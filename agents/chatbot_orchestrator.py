"""Conversational revision orchestrator for safe pipeline updates."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from core.action_registry import ControlledActionRegistry
from core.diff_utils import build_comparison_summary, diff_mapping, list_diff
from core.pipeline_state import PipelineState
from core.rerun_engine import DependencyAwareRerunEngine
from core.revision_history import RevisionHistoryManager
from core.revision_planner import RevisionPlan, RevisionPlanner


class ChatbotOrchestrator:
    """Interpret chat requests and route them into safe pipeline revisions."""

    def __init__(self) -> None:
        self._planner = RevisionPlanner()
        self._actions = ControlledActionRegistry()
        self._history = RevisionHistoryManager()
        self._reruns = DependencyAwareRerunEngine()

    async def handle_message(
        self,
        *,
        state: PipelineState,
        question: str,
        mode: str,
        config: Any,
        history: list[dict[str, str]],
        selection_context: Optional[dict[str, Any]],
        stage_runner: Callable[[str, Any], Awaitable[None]],
        response_builder: Optional[Callable[..., tuple[str, bool, str]]] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Handle a chat message as either a revision request or a standard explanation."""
        if state.dataset is None or not state.target_column:
            return {
                "answer": "Load a dataset and choose a target column before using the revision agent.",
                "llm_used": False,
                "response_mode": "unavailable",
                "revision": None,
            }

        if self._is_apply_request(question, mode) and state.pending_revision_plan:
            pending = self._planner.plan(
                state.pending_revision_plan["user_goal"],
                self._planner_state_payload(state),
                selection_context,
            )
            return await self._apply_plan(
                state=state,
                plan=pending,
                config=config,
                stage_runner=stage_runner,
                revision_reason=f"Applied pending revision: {pending.user_goal}",
                question=question,
                history=history,
                selection_context=selection_context,
                response_builder=response_builder,
                request_id=request_id,
            )

        plan = self._planner.plan(
            question,
            self._planner_state_payload(state),
            selection_context,
        )

        if plan.intent_type == "undo":
            return await self._undo_last_revision(
                state=state,
                config=config,
                stage_runner=stage_runner,
                question=question,
                history=history,
                selection_context=selection_context,
                response_builder=response_builder,
                request_id=request_id,
            )
        if plan.intent_type == "compare":
            return self._compare_runs(
                state=state,
                question=question,
                history=history,
                selection_context=selection_context,
                response_builder=response_builder,
                request_id=request_id,
            )
        if plan.intent_type == "explain" and plan.feature_name:
            return self._explain_feature_selection(
                state=state,
                question=question,
                feature_name=plan.feature_name,
                history=history,
                selection_context=selection_context,
                response_builder=response_builder,
                request_id=request_id,
            )
        if plan.intent_type == "other" or (not plan.actions and plan.intent_type != "rerun_stage"):
            revision = {
                "mode": "suggest",
                "plan": plan.to_dict(),
                "applied": False,
            }
            return self._build_response(
                fallback_answer=(
                    "I could not map that request to the supported safe revision actions yet. "
                    "Try asking about overfitting, missing values, feature inclusion or removal, model choice, recall, accuracy, undo, or compare."
                ),
                revision=revision,
                question=question,
                history=history,
                selection_context=selection_context,
                response_builder=response_builder,
                request_id=request_id,
                extra_context={
                    "response_kind": "revision_request_unmapped",
                    "plan": plan.to_dict(),
                },
            )

        if mode != "apply":
            state.pending_revision_plan = plan.to_dict()
            revision = {
                "mode": "suggest",
                "plan": plan.to_dict(),
                "applied": False,
            }
            return self._build_response(
                fallback_answer=self._build_suggestion_message(plan),
                revision=revision,
                question=question,
                history=history,
                selection_context=selection_context,
                response_builder=response_builder,
                request_id=request_id,
                extra_context={
                    "response_kind": "revision_suggestion",
                    "plan": plan.to_dict(),
                },
            )

        return await self._apply_plan(
            state=state,
            plan=plan,
            config=config,
            stage_runner=stage_runner,
            revision_reason=question,
            question=question,
            history=history,
            selection_context=selection_context,
            response_builder=response_builder,
            request_id=request_id,
        )

    def _planner_state_payload(self, state: PipelineState) -> dict[str, Any]:
        current = state.current_structured_state()
        current["known_features"] = state.known_features()
        return current

    def preview_plan(
        self,
        *,
        state: PipelineState,
        question: str,
        selection_context: Optional[dict[str, Any]],
    ) -> RevisionPlan:
        """Return the planner's best interpretation without applying anything."""
        return self._planner.plan(
            question,
            self._planner_state_payload(state),
            selection_context,
        )

    def _is_apply_request(self, question: str, mode: str) -> bool:
        return mode == "apply"

    async def _apply_plan(
        self,
        *,
        state: PipelineState,
        plan: RevisionPlan,
        config: Any,
        stage_runner: Callable[[str, Any], Awaitable[None]],
        revision_reason: str,
        question: str,
        history: list[dict[str, str]],
        selection_context: Optional[dict[str, Any]],
        response_builder: Optional[Callable[..., tuple[str, bool, str]]],
        request_id: Optional[str],
    ) -> dict[str, Any]:
        current_payload = self._planner_state_payload(state)
        applied = self._actions.apply_actions(
            current_state=current_payload,
            stage_configs=state.stage_configs,
            actions=plan.actions,
        )
        previous_run = state.revision_history[-1] if state.revision_history else None
        state.stage_configs = applied["stage_configs"]
        rerun_stages = await self._reruns.rerun(
            state=state,
            rerun_from_stage=plan.rerun_from_stage,
            config=config,
            stage_runner=stage_runner,
        )
        new_run = self._history.commit_run(
            state,
            revision_reason=revision_reason,
            changed_stages=applied["changed_stages"] or plan.target_stages,
            changed_configs=applied["config_diffs"],
        )
        state.pending_revision_plan = None

        comparison = None
        if previous_run is not None:
            comparison = self._build_comparison_payload(previous_run, new_run)

        revision = {
            "mode": "apply",
            "plan": plan.to_dict(),
            "applied": True,
            "rerun_stages": rerun_stages,
            "comparison": comparison,
            "run_id": new_run.run_id,
        }
        return self._build_response(
            fallback_answer=self._build_apply_message(plan, rerun_stages, comparison),
            revision=revision,
            question=question,
            history=history,
            selection_context=selection_context,
            response_builder=response_builder,
            request_id=request_id,
            extra_context={
                "response_kind": "revision_applied",
                "plan": plan.to_dict(),
                "rerun_stages": rerun_stages,
                "comparison": comparison,
                "new_run_id": new_run.run_id,
            },
        )

    async def _undo_last_revision(
        self,
        *,
        state: PipelineState,
        config: Any,
        stage_runner: Callable[[str, Any], Awaitable[None]],
        question: str,
        history: list[dict[str, str]],
        selection_context: Optional[dict[str, Any]],
        response_builder: Optional[Callable[..., tuple[str, bool, str]]],
        request_id: Optional[str],
    ) -> dict[str, Any]:
        previous = self._history.previous_run(state)
        current = state.revision_history[-1] if state.revision_history else None
        if previous is None or current is None:
            revision = {
                "mode": "apply",
                "intent_type": "undo",
                "applied": False,
            }
            return self._build_response(
                fallback_answer="There is no earlier revision to undo yet.",
                revision=revision,
                question=question,
                history=history,
                selection_context=selection_context,
                response_builder=response_builder,
                request_id=request_id,
                extra_context={
                    "response_kind": "revision_undo_unavailable",
                },
            )

        changed_stages = self._history.changed_stages_between(previous, current) or ["training"]
        self._history.restore_stage_configs(state, previous)
        rerun_from_stage = changed_stages[0]
        rerun_stages = await self._reruns.rerun(
            state=state,
            rerun_from_stage=rerun_from_stage,
            config=config,
            stage_runner=stage_runner,
        )
        new_run = self._history.commit_run(
            state,
            revision_reason=question,
            changed_stages=changed_stages,
            changed_configs={"undo_target_run_id": previous.run_id},
        )
        comparison = self._build_comparison_payload(current, new_run)
        revision = {
            "mode": "apply",
            "intent_type": "undo",
            "applied": True,
            "rerun_stages": rerun_stages,
            "comparison": comparison,
            "run_id": new_run.run_id,
        }
        return self._build_response(
            fallback_answer=(
                f"Restored the previous revision and reran {', '.join(rerun_stages)}. "
                f"{comparison['summary']}"
            ),
            revision=revision,
            question=question,
            history=history,
            selection_context=selection_context,
            response_builder=response_builder,
            request_id=request_id,
            extra_context={
                "response_kind": "revision_undone",
                "rerun_stages": rerun_stages,
                "comparison": comparison,
                "restored_run_id": previous.run_id,
                "new_run_id": new_run.run_id,
            },
        )

    def _compare_runs(
        self,
        *,
        state: PipelineState,
        question: str,
        history: list[dict[str, str]],
        selection_context: Optional[dict[str, Any]],
        response_builder: Optional[Callable[..., tuple[str, bool, str]]],
        request_id: Optional[str],
    ) -> dict[str, Any]:
        current = state.revision_history[-1] if state.revision_history else None
        previous = self._history.previous_run(state)
        if current is None or previous is None:
            revision = {
                "mode": "suggest",
                "intent_type": "compare",
                "applied": False,
            }
            return self._build_response(
                fallback_answer="I need at least two stored runs before I can compare revisions.",
                revision=revision,
                question=question,
                history=history,
                selection_context=selection_context,
                response_builder=response_builder,
                request_id=request_id,
                extra_context={
                    "response_kind": "revision_compare_unavailable",
                },
            )

        comparison = self._build_comparison_payload(previous, current)
        revision = {
            "mode": "suggest",
            "intent_type": "compare",
            "applied": False,
            "comparison": comparison,
        }
        return self._build_response(
            fallback_answer=comparison["summary"],
            revision=revision,
            question=question,
            history=history,
            selection_context=selection_context,
            response_builder=response_builder,
            request_id=request_id,
            extra_context={
                "response_kind": "revision_compare",
                "comparison": comparison,
            },
        )

    def _explain_feature_selection(
        self,
        *,
        state: PipelineState,
        question: str,
        feature_name: str,
        history: list[dict[str, str]],
        selection_context: Optional[dict[str, Any]],
        response_builder: Optional[Callable[..., tuple[str, bool, str]]],
        request_id: Optional[str],
    ) -> dict[str, Any]:
        features = state.stage_results.get("features", {}) or {}
        explanations = features.get("llm_explanations", {}) or {}
        feature_map = explanations.get("featureExplanations", {}) or {}
        dropped_map = explanations.get("droppedFeatureExplanations", {}) or {}
        score = (features.get("feature_scores", {}) or {}).get(feature_name)

        if feature_name in feature_map:
            answer = feature_map[feature_name]
            if score is not None:
                answer += f" Its current importance score is {float(score):.4f}."
            revision = {
                "mode": "suggest",
                "intent_type": "explain",
                "feature_name": feature_name,
                "applied": False,
            }
            return self._build_response(
                fallback_answer=answer,
                revision=revision,
                question=question,
                history=history,
                selection_context=selection_context,
                response_builder=response_builder,
                request_id=request_id,
                extra_context={
                    "response_kind": "feature_selection_explanation",
                    "feature_name": feature_name,
                    "feature_status": "selected",
                    "stored_explanation": feature_map[feature_name],
                    "importance_score": score,
                    "source_llm_used": bool(explanations.get("llmUsed", False)),
                },
            )

        if feature_name in dropped_map:
            revision = {
                "mode": "suggest",
                "intent_type": "explain",
                "feature_name": feature_name,
                "applied": False,
            }
            return self._build_response(
                fallback_answer=dropped_map[feature_name],
                revision=revision,
                question=question,
                history=history,
                selection_context=selection_context,
                response_builder=response_builder,
                request_id=request_id,
                extra_context={
                    "response_kind": "feature_selection_explanation",
                    "feature_name": feature_name,
                    "feature_status": "dropped",
                    "stored_explanation": dropped_map[feature_name],
                    "importance_score": score,
                    "source_llm_used": bool(explanations.get("llmUsed", False)),
                },
            )

        revision = {
            "mode": "suggest",
            "intent_type": "explain",
            "feature_name": feature_name,
            "applied": False,
        }
        return self._build_response(
            fallback_answer=f"I could not find a stored selection explanation for `{feature_name}` in the current run.",
            revision=revision,
            question=question,
            history=history,
            selection_context=selection_context,
            response_builder=response_builder,
            request_id=request_id,
            extra_context={
                "response_kind": "feature_selection_explanation_missing",
                "feature_name": feature_name,
            },
        )

    def _build_comparison_payload(self, previous: Any, current: Any) -> dict[str, Any]:
        config_diffs = diff_mapping(previous.stage_configs, current.stage_configs)
        metric_diffs = diff_mapping(previous.metrics, current.metrics)
        previous_features = previous.metrics.get("selected_features", []) or []
        current_features = current.metrics.get("selected_features", []) or []
        feature_diffs = list_diff(previous_features, current_features)
        changed_stages = [
            stage for stage, diff in config_diffs.items()
            if diff not in ({}, None)
        ] if isinstance(config_diffs, dict) else []
        return {
            "current_run_id": current.run_id,
            "previous_run_id": previous.run_id,
            "changed_stages": changed_stages,
            "config_diffs": config_diffs,
            "metric_diffs": metric_diffs,
            "feature_diffs": feature_diffs,
            "summary": build_comparison_summary(changed_stages, metric_diffs if isinstance(metric_diffs, dict) else {}, feature_diffs),
        }

    def _build_suggestion_message(self, plan: RevisionPlan) -> str:
        if plan.intent_type == "rerun_stage":
            return (
                f"I understood this as an explicit rerun request for `{plan.rerun_from_stage}`. "
                f"I would restart from `{plan.rerun_from_stage}` and rerun all downstream stages."
            )
        action_names = ", ".join(action["action_type"] for action in plan.actions)
        stages = ", ".join(plan.target_stages)
        return (
            f"I understood your request as `{plan.intent_type}`. "
            f"I would revise {stages}, apply `{action_names}`, and rerun from `{plan.rerun_from_stage}`. "
            f"Reason: {plan.reason} Send the same request in apply mode, or reply with `apply it`, to execute it."
        )

    def _build_apply_message(
        self,
        plan: RevisionPlan,
        rerun_stages: list[str],
        comparison: Optional[dict[str, Any]],
    ) -> str:
        if plan.intent_type == "rerun_stage":
            parts = [
                f"Restarted the pipeline from `{plan.rerun_from_stage}`.",
                f"Reran: {', '.join(rerun_stages)}.",
            ]
            if comparison:
                parts.append(comparison["summary"])
            else:
                parts.append("A new revision was stored.")
            return " ".join(parts)

        parts = [
            f"I revised `{', '.join(plan.target_stages)}` for `{plan.intent_type}`.",
            f"Reran: {', '.join(rerun_stages)}.",
        ]
        if comparison:
            parts.append(comparison["summary"])
        else:
            parts.append("A new revision was stored.")
        return " ".join(parts)

    def _build_response(
        self,
        *,
        fallback_answer: str,
        revision: Optional[dict[str, Any]],
        question: str,
        history: list[dict[str, str]],
        selection_context: Optional[dict[str, Any]],
        response_builder: Optional[Callable[..., tuple[str, bool, str]]],
        request_id: Optional[str],
        extra_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Generate the final chat answer, preferring the shared LLM responder."""
        if response_builder is not None:
            try:
                answer, llm_used, response_mode = response_builder(
                    question=question,
                    history=history,
                    selection_context=selection_context,
                    extra_context=extra_context,
                    request_id=request_id,
                )
                if answer and response_mode != "unavailable":
                    return {
                        "answer": answer,
                        "llm_used": llm_used,
                        "response_mode": response_mode,
                        "revision": revision,
                    }
            except Exception:
                pass

        return {
            "answer": fallback_answer,
            "llm_used": False,
            "response_mode": "structured",
            "revision": revision,
        }
