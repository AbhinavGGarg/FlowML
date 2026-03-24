"""Tests for the conversational pipeline revision agent."""

from __future__ import annotations

import pandas as pd
import pytest

from agents.chatbot_orchestrator import ChatbotOrchestrator
from core.action_registry import ControlledActionRegistry
from core.pipeline_state import PipelineState
from core.rerun_engine import DependencyAwareRerunEngine
from core.revision_history import RevisionHistoryManager
from core.revision_planner import RevisionPlanner


def build_seeded_state() -> PipelineState:
    """Create a runtime state with one completed baseline run."""
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [6, 5, 4, 3, 2, 1],
            "feature3": [10, 11, 12, 13, 14, 15],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    state = PipelineState()
    state.reset_for_dataset(
        df=df,
        dataset_path="uploads/test.csv",
        dataset_filename="test.csv",
        pipeline_id="baseline-run",
    )
    state.target_column = "target"
    state.update_pipeline_config(
        {
            "task_type": "classification",
            "test_size": 0.2,
            "random_state": 42,
        }
    )
    state.stage_results = {
        "analysis": {"row_count": len(df), "feature_count": 3},
        "preprocessing": {
            "dropped_columns": [],
            "missing_summary": {"strategy_used": "impute", "dropped_rows_count": 0},
        },
        "features": {
            "selected_features": ["feature1", "feature2"],
            "generated_features": [],
            "feature_scores": {"feature1": 0.64, "feature2": 0.36},
            "llm_explanations": {
                "llmUsed": False,
                "featureExplanations": {
                    "feature1": "feature1 stayed because it carried a strong direct signal.",
                },
                "droppedFeatureExplanations": {},
            },
        },
        "training": {
            "model_name": "RandomForest",
            "best_score": 0.78,
            "train_score": 0.95,
            "test_score": 0.75,
            "selected_features": ["feature1", "feature2"],
            "hyperparameters": {"max_depth": 10},
        },
        "evaluation": {
            "task_type": "classification",
            "accuracy": 0.75,
            "precision": 0.76,
            "recall": 0.70,
            "f1": 0.74,
            "deployment_decision": "iterate",
        },
        "results": {"model_path": "outputs/fake.pkl"},
        "explanation": {"summary": "Baseline explanation"},
    }
    state.stage_statuses = {stage: "completed" for stage in state.stage_statuses}
    RevisionHistoryManager().commit_run(
        state,
        revision_reason="Initial pipeline run",
        changed_stages=["analysis", "preprocessing", "feature_engineering", "training", "evaluation", "explainability"],
        changed_configs={},
    )
    return state


async def fake_stage_runner(stage: str, config) -> None:
    """Placeholder runner signature for type hints."""
    raise NotImplementedError


def build_fake_stage_runner(state: PipelineState):
    """Return a deterministic async stage runner for revision tests."""

    async def _runner(stage: str, config) -> None:
        state.stage_statuses[stage] = "completed"
        if stage == "preprocessing":
            strategy = state.stage_configs["preprocessing"]["missing_value_strategy"]
            state.stage_results["preprocessing"] = {
                "dropped_columns": [],
                "missing_summary": {
                    "strategy_used": strategy,
                    "dropped_rows_count": 0 if strategy == "impute" else 2,
                },
            }
        elif stage == "features":
            selected = ["feature1", "feature2"]
            include = state.stage_configs["feature_engineering"]["include_features"]
            exclude = set(state.stage_configs["feature_engineering"]["exclude_features"])
            for feature_name in include:
                if feature_name not in selected:
                    selected.append(feature_name)
            selected = [feature for feature in selected if feature not in exclude]
            state.stage_results["features"] = {
                "selected_features": selected,
                "generated_features": [],
                "feature_scores": {feature: 0.5 for feature in selected},
                "llm_explanations": {
                    "llmUsed": False,
                    "featureExplanations": {
                        feature: f"{feature} remained selected after the revision."
                        for feature in selected
                    },
                    "droppedFeatureExplanations": {},
                },
            }
        elif stage == "model_selection":
            state.stage_results["model_selection"] = {
                "selected_features": state.stage_results.get("features", {}).get("selected_features", ["feature1", "feature2"]),
                "task_type": state.task_type,
                "target_column": state.target_column,
                "top_candidates": [{"model_name": "RandomForest"}],
            }
        elif stage == "training":
            reduce_complexity = bool(state.stage_configs["training"]["reduce_complexity"])
            metric_priority = state.stage_configs["training"]["metric_priority"]
            selected_features = state.stage_results.get("features", {}).get("selected_features", ["feature1", "feature2"])
            state.stage_results["training"] = {
                "model_name": "RandomForest",
                "best_score": 0.82 if reduce_complexity else 0.78,
                "train_score": 0.88 if reduce_complexity else 0.95,
                "test_score": 0.82 if reduce_complexity else 0.75,
                "selected_features": selected_features,
                "hyperparameters": {"max_depth": 6 if reduce_complexity else 10},
                "metric_priority": metric_priority,
            }
        elif stage == "loss":
            training = state.stage_results["training"]
            state.stage_results["loss"] = {
                "train_loss": [1.0 - training["train_score"]],
                "val_loss": [1.0 - training["test_score"]],
                "best_epoch": 0,
            }
        elif stage == "evaluation":
            training = state.stage_results["training"]
            recall_focus = training.get("metric_priority") == "recall"
            state.stage_results["evaluation"] = {
                "task_type": "classification",
                "accuracy": training["test_score"],
                "precision": 0.80 if recall_focus else 0.76,
                "recall": 0.84 if recall_focus else 0.70,
                "f1": 0.82 if recall_focus else training["test_score"] - 0.01,
                "deployment_decision": "deploy" if training["test_score"] >= 0.8 else "iterate",
                "primary_metric": state.stage_configs["evaluation"]["primary_metric"] or "accuracy",
            }
        elif stage == "results":
            state.stage_results["results"] = {"model_path": "outputs/fake.pkl"}
            state.stage_results["explanation"] = {"summary": "Revision explanation"}

    return _runner


def test_revision_planner_maps_overfitting_request(monkeypatch) -> None:
    """The planner should map overfitting language into training-safe actions."""
    planner = RevisionPlanner()
    monkeypatch.setattr(planner._llm, "is_enabled", lambda: False)
    state = build_seeded_state()

    plan = planner.plan(
        "make the model less prone to overfitting",
        {**state.current_structured_state(), "known_features": state.known_features()},
    )

    assert plan.intent_type == "reduce_overfitting"
    assert plan.rerun_from_stage == "training"
    assert [action["action_type"] for action in plan.actions] == [
        "reduce_model_complexity",
        "increase_regularization",
    ]


def test_revision_planner_falls_back_to_phrase_heuristics_when_llm_plan_is_missing(monkeypatch) -> None:
    """If the planner LLM cannot map the request, deterministic heuristics should still work."""
    planner = RevisionPlanner()
    state = build_seeded_state()
    monkeypatch.setattr(planner._llm, "is_enabled", lambda: True)
    monkeypatch.setattr(planner, "_maybe_plan_with_llm", lambda *args, **kwargs: None)

    plan = planner.plan(
        "make the model less prone to overfitting",
        {**state.current_structured_state(), "known_features": state.known_features()},
    )

    assert plan.intent_type == "reduce_overfitting"
    assert plan.rerun_from_stage == "training"


def test_action_registry_updates_feature_configs() -> None:
    """Including a feature should update the feature stage config deterministically."""
    state = build_seeded_state()
    registry = ControlledActionRegistry()

    result = registry.apply_actions(
        current_state={**state.current_structured_state(), "known_features": state.known_features()},
        stage_configs=state.stage_configs,
        actions=[
            {
                "stage": "feature_engineering",
                "action_type": "include_feature",
                "params": {"feature_name": "feature3"},
            }
        ],
    )

    assert "feature_engineering" in result["changed_stages"]
    assert result["stage_configs"]["feature_engineering"]["include_features"] == ["feature3"]
    assert result["stage_configs"]["preprocessing"]["force_keep_columns"] == []


def test_revision_planner_keeps_feature_inclusion_at_feature_stage_when_column_is_available() -> None:
    """Feature inclusion should stay at feature engineering when preprocessing already kept the column."""
    planner = RevisionPlanner()
    state = build_seeded_state()

    plan = planner.plan(
        "use feature3 too",
        {**state.current_structured_state(), "known_features": state.known_features()},
    )

    assert plan.intent_type == "include_feature"
    assert plan.rerun_from_stage == "feature_engineering"
    assert plan.target_stages == ["feature_engineering"]
    assert plan.actions == [
        {
            "stage": "feature_engineering",
            "action_type": "include_feature",
            "params": {"feature_name": "feature3"},
        }
    ]


def test_revision_planner_moves_feature_inclusion_to_preprocessing_when_column_was_dropped() -> None:
    """Feature inclusion should restart from preprocessing when that stage removed the column."""
    planner = RevisionPlanner()
    state = build_seeded_state()
    state.stage_results["preprocessing"]["kept_feature_columns"] = ["feature1", "feature2"]
    state.stage_results["preprocessing"]["dropped_columns"] = [
        {"column": "feature3", "reason": "manual", "detail": "removed earlier"},
    ]

    plan = planner.plan(
        "use feature3 too",
        {**state.current_structured_state(), "known_features": state.known_features()},
    )

    assert plan.intent_type == "include_feature"
    assert plan.rerun_from_stage == "preprocessing"
    assert plan.target_stages == ["preprocessing", "feature_engineering"]
    assert plan.actions[0] == {
        "stage": "preprocessing",
        "action_type": "keep_column",
        "params": {"feature_name": "feature3"},
    }
    assert plan.actions[1] == {
        "stage": "feature_engineering",
        "action_type": "include_feature",
        "params": {"feature_name": "feature3"},
    }


def test_revision_planner_maps_run_without_feature_to_preprocessing_rerun() -> None:
    """A 'run without feature' request should drop the raw column and its engineered dependents."""
    planner = RevisionPlanner()
    state = build_seeded_state()
    state.stage_results["features"] = {
        "selected_features": ["feature1", "feature2", "feature1__mul__feature2"],
        "generated_features": ["feature1__mul__feature2"],
        "dropped_columns": [],
        "feature_scores": {
            "feature1": 0.64,
            "feature2": 0.36,
            "feature1__mul__feature2": 0.18,
        },
        "llm_explanations": {
            "llmUsed": False,
            "featureExplanations": {},
            "droppedFeatureExplanations": {},
        },
    }

    plan = planner.plan(
        "run the model without feature1",
        {**state.current_structured_state(), "known_features": state.known_features()},
    )

    assert plan.intent_type == "exclude_feature"
    assert plan.rerun_from_stage == "preprocessing"
    assert plan.target_stages == ["preprocessing", "feature_engineering"]
    assert plan.actions[0] == {
        "stage": "preprocessing",
        "action_type": "drop_column",
        "params": {"feature_name": "feature1"},
    }
    assert {
        "stage": "feature_engineering",
        "action_type": "exclude_feature",
        "params": {"feature_name": "feature1"},
    } in plan.actions
    assert {
        "stage": "feature_engineering",
        "action_type": "force_drop_engineered_feature",
        "params": {"feature_name": "feature1__mul__feature2"},
    } in plan.actions


def test_rerun_engine_resolves_downstream_stages() -> None:
    """Changing preprocessing should rerun that stage and everything after it."""
    engine = DependencyAwareRerunEngine()

    assert engine.resolve_concrete_stages("preprocessing") == [
        "preprocessing",
        "features",
        "model_selection",
        "training",
        "loss",
        "evaluation",
        "results",
    ]


def test_revision_planner_maps_explicit_stage_rerun_request() -> None:
    """Explicit stage rerun language should map directly to the requested stage."""
    planner = RevisionPlanner()
    state = build_seeded_state()

    plan = planner.plan(
        "rerun evaluation",
        {**state.current_structured_state(), "known_features": state.known_features()},
    )

    assert plan.intent_type == "rerun_stage"
    assert plan.rerun_from_stage == "evaluation"
    assert plan.target_stages == ["evaluation"]
    assert plan.actions == []


def test_rerun_engine_resolves_evaluation_downstream_only() -> None:
    """Evaluation reruns should not reset or rerun earlier stages."""
    engine = DependencyAwareRerunEngine()

    assert engine.resolve_concrete_stages("evaluation") == [
        "evaluation",
        "results",
    ]


@pytest.mark.asyncio
async def test_chatbot_orchestrator_supports_suggest_apply_compare_and_undo(monkeypatch) -> None:
    """The revision orchestrator should support the core conversational workflow."""
    state = build_seeded_state()
    orchestrator = ChatbotOrchestrator()
    monkeypatch.setattr(orchestrator._planner._llm, "is_enabled", lambda: False)
    runner = build_fake_stage_runner(state)

    def fake_response_builder(*, extra_context=None, **kwargs):
        response_kind = str((extra_context or {}).get("response_kind") or "unknown")
        return (f"llm::{response_kind}", True, "llm")

    suggest = await orchestrator.handle_message(
        state=state,
        question="make the model less prone to overfitting",
        mode="suggest",
        config=type("Config", (), state.pipeline_config)(),
        history=[],
        selection_context=None,
        stage_runner=runner,
        response_builder=fake_response_builder,
        request_id="rev-test",
    )
    assert suggest["answer"] == "llm::revision_suggestion"
    assert suggest["llm_used"] is True
    assert suggest["response_mode"] == "llm"
    assert suggest["revision"]["mode"] == "suggest"
    assert suggest["revision"]["applied"] is False
    assert state.pending_revision_plan is not None

    applied = await orchestrator.handle_message(
        state=state,
        question="apply it",
        mode="apply",
        config=type("Config", (), state.pipeline_config)(),
        history=[],
        selection_context=None,
        stage_runner=runner,
        response_builder=fake_response_builder,
        request_id="rev-test",
    )
    assert applied["answer"] == "llm::revision_applied"
    assert applied["llm_used"] is True
    assert applied["response_mode"] == "llm"
    assert applied["revision"]["applied"] is True
    assert state.stage_results["training"]["test_score"] == pytest.approx(0.82)
    assert len(state.revision_history) == 2

    compared = await orchestrator.handle_message(
        state=state,
        question="compare this with the previous run",
        mode="suggest",
        config=type("Config", (), state.pipeline_config)(),
        history=[],
        selection_context=None,
        stage_runner=runner,
        response_builder=fake_response_builder,
        request_id="rev-test",
    )
    assert compared["answer"] == "llm::revision_compare"
    assert compared["llm_used"] is True
    assert compared["revision"]["comparison"]["current_run_id"] == state.current_run_id

    undone = await orchestrator.handle_message(
        state=state,
        question="undo the last change",
        mode="suggest",
        config=type("Config", (), state.pipeline_config)(),
        history=[],
        selection_context=None,
        stage_runner=runner,
        response_builder=fake_response_builder,
        request_id="rev-test",
    )
    assert undone["answer"] == "llm::revision_undone"
    assert undone["llm_used"] is True
    assert undone["revision"]["applied"] is True
    assert state.stage_results["training"]["test_score"] == pytest.approx(0.75)
    assert len(state.revision_history) == 3


@pytest.mark.asyncio
async def test_chatbot_orchestrator_applies_explicit_evaluation_rerun(monkeypatch) -> None:
    """Explicit evaluation reruns should only execute evaluation and explainability work."""
    state = build_seeded_state()
    orchestrator = ChatbotOrchestrator()
    monkeypatch.setattr(orchestrator._planner._llm, "is_enabled", lambda: False)
    executed_stages: list[str] = []
    base_runner = build_fake_stage_runner(state)

    async def runner(stage: str, config) -> None:
        executed_stages.append(stage)
        await base_runner(stage, config)

    response = await orchestrator.handle_message(
        state=state,
        question="rerun evaluation",
        mode="apply",
        config=type("Config", (), state.pipeline_config)(),
        history=[],
        selection_context=None,
        stage_runner=runner,
        response_builder=None,
        request_id="rev-test",
    )

    assert response["revision"]["applied"] is True
    assert response["revision"]["plan"]["rerun_from_stage"] == "evaluation"
    assert response["revision"]["rerun_stages"] == ["evaluation", "results"]
    assert executed_stages == ["evaluation", "results"]
    assert state.stage_logs["evaluation"][-2].startswith("Revision rerun requested from canonical stage")


@pytest.mark.asyncio
async def test_chatbot_orchestrator_explains_selected_feature(monkeypatch) -> None:
    """Feature-selection explanation requests should use the shared LLM response path."""
    state = build_seeded_state()
    orchestrator = ChatbotOrchestrator()
    monkeypatch.setattr(orchestrator._planner._llm, "is_enabled", lambda: False)

    def fake_response_builder(*, extra_context=None, **kwargs):
        feature_name = str((extra_context or {}).get("feature_name") or "")
        feature_status = str((extra_context or {}).get("feature_status") or "")
        importance_score = extra_context.get("importance_score")
        return (f"{feature_name}:{feature_status}:{importance_score}", True, "llm")

    response = await orchestrator.handle_message(
        state=state,
        question="explain why feature1 was selected",
        mode="suggest",
        config=type("Config", (), state.pipeline_config)(),
        history=[],
        selection_context=None,
        stage_runner=build_fake_stage_runner(state),
        response_builder=fake_response_builder,
        request_id="rev-test",
    )

    assert response["revision"]["intent_type"] == "explain"
    assert response["answer"] == "feature1:selected:0.64"
    assert response["llm_used"] is True
    assert response["response_mode"] == "llm"
