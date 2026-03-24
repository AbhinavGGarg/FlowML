"""Intent mapping and revision planning for conversational pipeline edits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Optional

from config import settings
from core.pipeline_state import CANONICAL_STAGE_ORDER, normalize_stage_name
from utils.openrouter_client import OpenRouterClient


@dataclass
class RevisionPlan:
    """Structured output from the revision planner."""

    user_goal: str
    intent_type: str
    target_stages: list[str]
    actions: list[dict[str, Any]]
    rerun_from_stage: str
    reason: str
    confidence: str
    feature_name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a dict representation."""
        return asdict(self)


class RevisionPlanner:
    """Map grounded user requests to safe deterministic actions."""

    def __init__(self) -> None:
        self._llm = OpenRouterClient("RevisionPlanner", model_name=settings.chat_model_name)

    def plan(
        self,
        user_request: str,
        current_state: dict[str, Any],
        selection_context: Optional[dict[str, Any]] = None,
    ) -> RevisionPlan:
        """Interpret a revision request against the current pipeline state."""
        text = user_request.strip()
        lower = text.lower()
        feature_name = self._extract_feature_name(text, current_state, selection_context)

        if any(phrase in lower for phrase in ("undo", "revert", "go back")):
            return self._finalize_plan(RevisionPlan(
                user_goal=text,
                intent_type="undo",
                target_stages=[],
                actions=[],
                rerun_from_stage="training",
                reason="The request asks to restore the previous pipeline revision.",
                confidence="high",
            ), current_state)

        if "compare" in lower and "previous" in lower:
            return self._finalize_plan(RevisionPlan(
                user_goal=text,
                intent_type="compare",
                target_stages=[],
                actions=[],
                rerun_from_stage="evaluation",
                reason="The request asks for a comparison against the last stored revision.",
                confidence="high",
            ), current_state)

        if "why" in lower and "selected" in lower and feature_name:
            return self._finalize_plan(RevisionPlan(
                user_goal=text,
                intent_type="explain",
                target_stages=["feature_engineering"],
                actions=[],
                rerun_from_stage="feature_engineering",
                reason="The request is asking for an explanation of the feature-selection decision.",
                confidence="high",
                feature_name=feature_name,
            ), current_state)

        explicit_stage = self._detect_explicit_stage_request(text)
        if explicit_stage:
            return self._finalize_plan(
                self._build_stage_rerun_plan(text, explicit_stage),
                current_state,
            )

        if any(phrase in lower for phrase in ("overfitting", "overfit", "less prone to overfitting")):
            return self._finalize_plan(RevisionPlan(
                user_goal=text,
                intent_type="reduce_overfitting",
                target_stages=["training"],
                actions=[
                    {"stage": "training", "action_type": "reduce_model_complexity", "params": {}},
                    {"stage": "training", "action_type": "increase_regularization", "params": {}},
                ],
                rerun_from_stage="training",
                reason="Overfitting is primarily controlled by the training setup, so the safest first revision is to reduce model complexity and increase regularization.",
                confidence="high",
            ), current_state)

        if feature_name and self._is_feature_inclusion_request(lower):
            return self._finalize_plan(
                self._build_feature_inclusion_plan(
                    user_goal=text,
                    current_state=current_state,
                    feature_name=feature_name,
                    reason="Feature inclusion belongs to the feature-engineering stage, unless preprocessing previously removed the requested column.",
                    confidence="high",
                ),
                current_state,
            )

        if feature_name and self._is_feature_exclusion_request(lower):
            return self._finalize_plan(self._build_feature_exclusion_plan(
                user_goal=text,
                current_state=current_state,
                feature_name=feature_name,
                reason=(
                    "Removing a feature changes the model inputs, so the pipeline should drop that column, "
                    "remove any dependent engineered features, and rerun downstream stages."
                ),
                confidence="high",
            ), current_state)

        if "missing" in lower and any(phrase in lower for phrase in ("don't drop", "dont drop", "keep rows", "do not drop")):
            return self._finalize_plan(RevisionPlan(
                user_goal=text,
                intent_type="change_preprocessing",
                target_stages=["preprocessing"],
                actions=[
                    {"stage": "preprocessing", "action_type": "protect_rows_from_drop", "params": {"enabled": True}},
                    {"stage": "preprocessing", "action_type": "change_missing_value_strategy", "params": {"strategy": "impute"}},
                ],
                rerun_from_stage="preprocessing",
                reason="Missing-value row dropping happens during preprocessing, so the revision should start there.",
                confidence="high",
            ), current_state)

        if "missing" in lower:
            return self._finalize_plan(RevisionPlan(
                user_goal=text,
                intent_type="change_preprocessing",
                target_stages=["preprocessing"],
                actions=[
                    {"stage": "preprocessing", "action_type": "change_missing_value_strategy", "params": {"strategy": "impute"}},
                ],
                rerun_from_stage="preprocessing",
                reason="The request refers to missing-value behavior, which is controlled in preprocessing.",
                confidence="medium",
            ), current_state)

        if any(phrase in lower for phrase in ("easier to explain", "more explainable", "different model", "try another model", "try a model")):
            return self._finalize_plan(RevisionPlan(
                user_goal=text,
                intent_type="switch_model",
                target_stages=["training"],
                actions=[
                    {"stage": "training", "action_type": "switch_model_family", "params": {"model_family": "linear"}},
                ],
                rerun_from_stage="training",
                reason="Model choice is a training-level revision, and a linear family is the safest interpretable fallback in this app.",
                confidence="medium",
            ), current_state)

        if "recall" in lower:
            return self._finalize_plan(RevisionPlan(
                user_goal=text,
                intent_type="improve_metric",
                target_stages=["training", "evaluation"],
                actions=[
                    {"stage": "training", "action_type": "enable_class_weights", "params": {"enabled": True}},
                    {"stage": "training", "action_type": "change_metric_priority", "params": {"metric": "recall"}},
                    {"stage": "training", "action_type": "retune_hyperparameters", "params": {"enabled": True}},
                    {"stage": "evaluation", "action_type": "change_primary_metric", "params": {"metric": "recall"}},
                ],
                rerun_from_stage="training",
                reason="Recall-oriented improvements are safest when expressed as a training objective plus an evaluation focus.",
                confidence="high",
            ), current_state)

        if "accuracy" in lower:
            return self._finalize_plan(RevisionPlan(
                user_goal=text,
                intent_type="improve_metric",
                target_stages=["training", "evaluation"],
                actions=[
                    {"stage": "training", "action_type": "change_metric_priority", "params": {"metric": "accuracy"}},
                    {"stage": "training", "action_type": "retune_hyperparameters", "params": {"enabled": True}},
                    {"stage": "evaluation", "action_type": "change_primary_metric", "params": {"metric": "accuracy"}},
                ],
                rerun_from_stage="training",
                reason="Accuracy improvements are handled most safely by retuning the training objective and reevaluating the resulting model.",
                confidence="high",
            ), current_state)

        llm_plan = self._maybe_plan_with_llm(text, current_state, selection_context)
        if llm_plan:
            return self._finalize_plan(llm_plan, current_state)

        return self._finalize_plan(RevisionPlan(
            user_goal=text,
            intent_type="other",
            target_stages=[],
            actions=[],
            rerun_from_stage="evaluation",
            reason="The request did not map cleanly onto the supported deterministic revision actions.",
            confidence="low",
            feature_name=feature_name,
        ), current_state)

    def _finalize_plan(
        self,
        plan: RevisionPlan,
        current_state: dict[str, Any],
    ) -> RevisionPlan:
        """Normalize stage names and compute the minimal safe rerun boundary."""
        normalized_targets = list(dict.fromkeys(
            normalize_stage_name(stage)
            for stage in plan.target_stages
            if normalize_stage_name(stage) in CANONICAL_STAGE_ORDER
        ))
        normalized_actions = [
            {
                **action,
                "stage": normalize_stage_name(str(action.get("stage") or "").strip()),
            }
            for action in plan.actions
        ]
        normalized_rerun_stage = self._resolve_rerun_from_stage(
            plan=plan,
            current_state=current_state,
            normalized_targets=normalized_targets,
            normalized_actions=normalized_actions,
        )

        final_targets = normalized_targets
        if plan.intent_type == "rerun_stage" and not final_targets:
            final_targets = [normalized_rerun_stage]

        return RevisionPlan(
            user_goal=plan.user_goal,
            intent_type=plan.intent_type,
            target_stages=final_targets,
            actions=normalized_actions,
            rerun_from_stage=normalized_rerun_stage,
            reason=plan.reason,
            confidence=plan.confidence,
            feature_name=plan.feature_name,
        )

    def _resolve_rerun_from_stage(
        self,
        *,
        plan: RevisionPlan,
        current_state: dict[str, Any],
        normalized_targets: list[str],
        normalized_actions: list[dict[str, Any]],
    ) -> str:
        """Compute the earliest safe canonical rerun stage for a plan."""
        explicit_stage = self._detect_explicit_stage_request(plan.user_goal)
        if explicit_stage:
            return explicit_stage

        requested_stage = normalize_stage_name(plan.rerun_from_stage)
        if plan.intent_type in {"undo", "compare", "explain", "other"}:
            return requested_stage
        if plan.intent_type == "rerun_stage":
            return normalized_targets[0] if normalized_targets else requested_stage
        if plan.intent_type == "include_feature" and plan.feature_name:
            return (
                "preprocessing"
                if self._feature_requires_preprocessing_rerun(plan.feature_name, current_state)
                else "feature_engineering"
            )

        ranked_stages = [
            stage for stage in [
                *normalized_targets,
                *(normalize_stage_name(str(action.get("stage") or "")) for action in normalized_actions),
            ]
            if stage in CANONICAL_STAGE_ORDER
        ]
        if ranked_stages:
            return min(ranked_stages, key=self._stage_order_index)
        return requested_stage

    def _stage_order_index(self, stage: str) -> int:
        canonical_stage = normalize_stage_name(stage)
        if canonical_stage not in CANONICAL_STAGE_ORDER:
            return len(CANONICAL_STAGE_ORDER)
        return CANONICAL_STAGE_ORDER.index(canonical_stage)

    def _build_stage_rerun_plan(self, user_goal: str, stage: str) -> RevisionPlan:
        """Build a no-config-change plan that reruns a specific stage and downstream work."""
        canonical_stage = normalize_stage_name(stage)
        return RevisionPlan(
            user_goal=user_goal,
            intent_type="rerun_stage",
            target_stages=[canonical_stage],
            actions=[],
            rerun_from_stage=canonical_stage,
            reason=f"The request explicitly asks to rerun the `{canonical_stage}` stage and its downstream dependencies.",
            confidence="high",
        )

    def _build_feature_inclusion_plan(
        self,
        *,
        user_goal: str,
        current_state: dict[str, Any],
        feature_name: str,
        reason: str,
        confidence: str,
    ) -> RevisionPlan:
        """Build a grounded feature-inclusion plan with the minimal safe rerun boundary."""
        rerun_from_stage = "feature_engineering"
        target_stages = ["feature_engineering"]
        actions: list[dict[str, Any]] = [
            {
                "stage": "feature_engineering",
                "action_type": "include_feature",
                "params": {"feature_name": feature_name},
            }
        ]

        if self._feature_requires_preprocessing_rerun(feature_name, current_state):
            rerun_from_stage = "preprocessing"
            target_stages = ["preprocessing", "feature_engineering"]
            actions.insert(0, {
                "stage": "preprocessing",
                "action_type": "keep_column",
                "params": {"feature_name": feature_name},
            })

        return RevisionPlan(
            user_goal=user_goal,
            intent_type="include_feature",
            target_stages=target_stages,
            actions=actions,
            rerun_from_stage=rerun_from_stage,
            reason=reason,
            confidence=confidence,
            feature_name=feature_name,
        )

    def _feature_requires_preprocessing_rerun(
        self,
        feature_name: str,
        current_state: dict[str, Any],
    ) -> bool:
        """Return whether the requested raw feature is currently unavailable after preprocessing."""
        dataset_columns = {
            str(column)
            for column in ((current_state.get("dataset") or {}).get("column_names") or [])
            if str(column)
        }
        if feature_name not in dataset_columns:
            return False

        preprocessing_output = ((current_state.get("stage_outputs") or {}).get("preprocessing")) or {}
        kept_feature_columns = preprocessing_output.get("kept_feature_columns", [])
        if isinstance(kept_feature_columns, list) and kept_feature_columns and feature_name not in {
            str(column) for column in kept_feature_columns
        }:
            return True

        dropped_columns = preprocessing_output.get("dropped_columns", [])
        for item in dropped_columns if isinstance(dropped_columns, list) else []:
            if isinstance(item, dict) and str(item.get("column") or "") == feature_name:
                return True
        return False

    def _detect_explicit_stage_request(self, text: str) -> Optional[str]:
        """Detect explicit requests to rerun a specific stage."""
        stage_patterns = [
            ("analysis", r"(?:analysis|analy[sz]e(?:sis)?)"),
            ("preprocessing", r"(?:preprocessing|preprocess(?:ing)?|data cleaning)"),
            ("feature_engineering", r"(?:feature engineering|feature selection|features?)"),
            ("training", r"(?:training|model training|modeling|train(?:ing)?)"),
            ("evaluation", r"(?:evaluation|evaluate|validation|metrics)"),
            ("explainability", r"(?:explainability|explanations?|results)"),
        ]
        verbs = r"(?:rerun|re-run|restart|redo|recompute|refresh)"
        lowered = text.lower()
        for stage, pattern in stage_patterns:
            if re.search(rf"\b{verbs}\b[\w\s-]*\b{pattern}\b", lowered):
                return stage
        return None

    def _extract_feature_name(
        self,
        text: str,
        current_state: dict[str, Any],
        selection_context: Optional[dict[str, Any]],
    ) -> Optional[str]:
        candidates = [str(item) for item in current_state.get("known_features", [])]
        lower_map = {candidate.lower(): candidate for candidate in candidates}

        quoted = self._quoted_fragment(text)
        if quoted and quoted.lower() in lower_map:
            return lower_map[quoted.lower()]

        lowered = text.lower()
        for candidate in sorted(candidates, key=len, reverse=True):
            if candidate.lower() in lowered:
                return candidate

        if selection_context:
            selected = str(selection_context.get("text") or "").strip()
            if selected.lower() in lower_map:
                return lower_map[selected.lower()]
        return None

    def _quoted_fragment(self, text: str) -> Optional[str]:
        quote_pairs = [('"', '"'), ("'", "'")]
        for left, right in quote_pairs:
            if left not in text or right not in text:
                continue
            first = text.find(left)
            last = text.find(right, first + 1)
            if first >= 0 and last > first:
                return text[first + 1:last].strip()
        return None

    def _maybe_plan_with_llm(
        self,
        user_request: str,
        current_state: dict[str, Any],
        selection_context: Optional[dict[str, Any]] = None,
    ) -> Optional[RevisionPlan]:
        """Use the LLM as the primary classifier for revision intent."""
        if not self._llm.is_enabled():
            return None
        feature_name = self._extract_feature_name(user_request, current_state, selection_context)

        payload = {
            "request": user_request,
            "task_type": current_state.get("task_type"),
            "known_features": current_state.get("known_features", [])[:40],
            "selected_text": str(selection_context.get("text") or "").strip() if selection_context else None,
            "supported_intents": [
                "reduce_overfitting",
                "include_feature",
                "exclude_feature",
                "change_preprocessing",
                "switch_model",
                "improve_metric",
                "explain",
                "undo",
                "compare",
                "other",
            ],
        }
        try:
            response = self._llm.generate_json(
                "You classify pipeline revision requests. Return JSON with keys intent_type, feature_name, metric, and confidence. Use only the provided supported_intents.",
                f"Revision request context: {payload}",
                temperature=0.0,
                max_tokens=300,
            )
        except Exception:
            return None

        intent = str(response.get("intent_type") or "").strip()
        feature_name = str(response.get("feature_name") or "").strip() or feature_name
        metric = str(response.get("metric") or "").strip() or None
        confidence = str(response.get("confidence") or "low").strip().lower()
        normalized_confidence = confidence if confidence in {"high", "medium", "low"} else "low"

        if intent == "include_feature" and feature_name:
            return self._build_feature_inclusion_plan(
                user_goal=user_request,
                current_state=current_state,
                feature_name=feature_name,
                reason="The request appears to be asking for a feature to be included.",
                confidence=normalized_confidence,
            )

        if intent == "exclude_feature" and feature_name:
            return self._build_feature_exclusion_plan(
                user_goal=user_request,
                current_state=current_state,
                feature_name=feature_name,
                reason="The request appears to be asking for a feature to be excluded from the rerun.",
                confidence=normalized_confidence,
            )

        if intent == "improve_metric" and metric:
            return RevisionPlan(
                user_goal=user_request,
                intent_type="improve_metric",
                target_stages=["training", "evaluation"],
                actions=[
                    {"stage": "training", "action_type": "change_metric_priority", "params": {"metric": metric}},
                    {"stage": "evaluation", "action_type": "change_primary_metric", "params": {"metric": metric}},
                ],
                rerun_from_stage="training",
                reason="The request appears to be metric-focused, so the safest deterministic response is to retune training and reevaluation around that metric.",
                confidence=normalized_confidence,
                feature_name=feature_name,
            )
        if intent == "reduce_overfitting":
            return RevisionPlan(
                user_goal=user_request,
                intent_type="reduce_overfitting",
                target_stages=["training"],
                actions=[
                    {"stage": "training", "action_type": "reduce_model_complexity", "params": {}},
                    {"stage": "training", "action_type": "increase_regularization", "params": {}},
                ],
                rerun_from_stage="training",
                reason="The request appears focused on overfitting, so the safest first revision is to simplify the training setup.",
                confidence=normalized_confidence,
                feature_name=feature_name,
            )
        if intent == "change_preprocessing":
            return RevisionPlan(
                user_goal=user_request,
                intent_type="change_preprocessing",
                target_stages=["preprocessing"],
                actions=[
                    {"stage": "preprocessing", "action_type": "change_missing_value_strategy", "params": {"strategy": "impute"}},
                ],
                rerun_from_stage="preprocessing",
                reason="The request appears to be about preprocessing behavior, so the revision should begin there.",
                confidence=normalized_confidence,
                feature_name=feature_name,
            )
        if intent == "switch_model":
            return RevisionPlan(
                user_goal=user_request,
                intent_type="switch_model",
                target_stages=["training"],
                actions=[
                    {"stage": "training", "action_type": "switch_model_family", "params": {"model_family": "linear"}},
                ],
                rerun_from_stage="training",
                reason="The request appears to be asking for a different model family.",
                confidence=normalized_confidence,
                feature_name=feature_name,
            )
        if intent == "explain" and feature_name:
            return RevisionPlan(
                user_goal=user_request,
                intent_type="explain",
                target_stages=["feature_engineering"],
                actions=[],
                rerun_from_stage="feature_engineering",
                reason="The request appears to ask for a feature-selection explanation.",
                confidence=normalized_confidence,
                feature_name=feature_name,
            )
        if intent == "undo":
            return RevisionPlan(
                user_goal=user_request,
                intent_type="undo",
                target_stages=[],
                actions=[],
                rerun_from_stage="training",
                reason="The request appears to ask for the previous revision to be restored.",
                confidence=normalized_confidence,
                feature_name=feature_name,
            )
        if intent == "compare":
            return RevisionPlan(
                user_goal=user_request,
                intent_type="compare",
                target_stages=[],
                actions=[],
                rerun_from_stage="evaluation",
                reason="The request appears to ask for a comparison with a previous revision.",
                confidence=normalized_confidence,
                feature_name=feature_name,
            )
        return None

    def _is_feature_exclusion_request(self, lowered_request: str) -> bool:
        """Return whether the request is asking to rerun without a feature."""
        markers = (
            "without ",
            "run without",
            "train without",
            "model without",
            "remove this feature",
            "exclude feature",
            "drop feature",
            "remove feature",
            "don't use",
            "dont use",
            "leave out",
            "take out",
        )
        return any(marker in lowered_request for marker in markers)

    def _is_feature_inclusion_request(self, lowered_request: str) -> bool:
        """Return whether the request is asking to add or keep a feature."""
        if self._is_feature_exclusion_request(lowered_request):
            return False
        explicit_markers = (
            "use this feature",
            "include feature",
            "use this too",
            "add feature",
            "add this feature",
        )
        if any(marker in lowered_request for marker in explicit_markers):
            return True
        regex_markers = (
            r"\binclude\b",
            r"\badd\b",
            r"\buse\b.*\b(?:too|also|again|as well)\b",
        )
        return any(re.search(pattern, lowered_request) for pattern in regex_markers)

    def _build_feature_exclusion_plan(
        self,
        *,
        user_goal: str,
        current_state: dict[str, Any],
        feature_name: str,
        reason: str,
        confidence: str,
    ) -> RevisionPlan:
        """Build a grounded feature-exclusion revision plan."""
        dataset_columns = {
            str(column)
            for column in ((current_state.get("dataset") or {}).get("column_names") or [])
            if str(column)
        }
        normalized_feature = str(feature_name)
        dependent_engineered = self._find_dependent_engineered_features(normalized_feature, current_state)
        is_raw_column = normalized_feature in dataset_columns
        is_engineered_feature = self._is_engineered_feature(normalized_feature, current_state)

        actions: list[dict[str, Any]] = []
        target_stages: list[str] = []
        rerun_from_stage = "feature_engineering"

        if is_raw_column:
            target_stages.append("preprocessing")
            rerun_from_stage = "preprocessing"
            actions.append({
                "stage": "preprocessing",
                "action_type": "drop_column",
                "params": {"feature_name": normalized_feature},
            })

        target_stages.append("feature_engineering")
        actions.append({
            "stage": "feature_engineering",
            "action_type": "exclude_feature",
            "params": {"feature_name": normalized_feature},
        })

        if is_engineered_feature:
            actions.append({
                "stage": "feature_engineering",
                "action_type": "force_drop_engineered_feature",
                "params": {"feature_name": normalized_feature},
            })

        seen_engineered: set[str] = set()
        for dependent_feature in dependent_engineered:
            if dependent_feature in seen_engineered:
                continue
            seen_engineered.add(dependent_feature)
            actions.append({
                "stage": "feature_engineering",
                "action_type": "force_drop_engineered_feature",
                "params": {"feature_name": dependent_feature},
            })

        return RevisionPlan(
            user_goal=user_goal,
            intent_type="exclude_feature",
            target_stages=list(dict.fromkeys(target_stages)),
            actions=actions,
            rerun_from_stage=rerun_from_stage,
            reason=reason,
            confidence=confidence,
            feature_name=normalized_feature,
        )

    def _find_dependent_engineered_features(
        self,
        feature_name: str,
        current_state: dict[str, Any],
    ) -> list[str]:
        """Return engineered features that appear to depend on a raw feature."""
        feature_outputs = (
            ((current_state.get("stage_outputs") or {}).get("feature_engineering"))
            or {}
        )
        candidates: list[str] = []
        for key in ("generated_features", "selected_features", "dropped_columns"):
            values = feature_outputs.get(key, [])
            if isinstance(values, list):
                candidates.extend(str(value) for value in values if str(value))

        pattern = re.compile(rf"(^|[^a-zA-Z0-9]){re.escape(feature_name)}([^a-zA-Z0-9]|$)", re.IGNORECASE)
        dependent = [
            candidate for candidate in candidates
            if candidate != feature_name
            and self._looks_engineered(candidate)
            and pattern.search(candidate)
        ]
        return list(dict.fromkeys(dependent))

    def _is_engineered_feature(
        self,
        feature_name: str,
        current_state: dict[str, Any],
    ) -> bool:
        """Return whether a feature name appears to be engineered in the current state."""
        feature_outputs = (
            ((current_state.get("stage_outputs") or {}).get("feature_engineering"))
            or {}
        )
        generated = feature_outputs.get("generated_features", [])
        return isinstance(generated, list) and feature_name in {str(value) for value in generated}

    def _looks_engineered(self, feature_name: str) -> bool:
        """Heuristic for generated feature names."""
        markers = ("__", "_mul_", "_div_", "_plus_", "_minus_", "pca_", "interaction")
        return any(marker in feature_name.lower() for marker in markers)
