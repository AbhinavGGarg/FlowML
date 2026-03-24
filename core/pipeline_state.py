"""Structured pipeline state for runtime execution and revision tracking."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

import pandas as pd

from core.diff_utils import public_value


CANONICAL_STAGE_ORDER = [
    "analysis",
    "preprocessing",
    "feature_engineering",
    "training",
    "evaluation",
    "explainability",
]

CONCRETE_STAGE_ORDER = [
    "analysis",
    "preprocessing",
    "features",
    "model_selection",
    "training",
    "loss",
    "evaluation",
    "results",
]

CANONICAL_TO_CONCRETE_STAGE = {
    "analysis": "analysis",
    "preprocessing": "preprocessing",
    "feature_engineering": "features",
    "training": "training",
    "evaluation": "evaluation",
    "explainability": "results",
}

STAGE_NAME_ALIASES = {
    "analysis": "analysis",
    "analyze": "analysis",
    "preprocess": "preprocessing",
    "preprocessing": "preprocessing",
    "data_cleaning": "preprocessing",
    "feature_engineering": "feature_engineering",
    "feature_selection": "feature_engineering",
    "features": "feature_engineering",
    "training": "training",
    "train": "training",
    "modeling": "training",
    "model_training": "training",
    "evaluation": "evaluation",
    "evaluate": "evaluation",
    "metrics": "evaluation",
    "validation": "evaluation",
    "explainability": "explainability",
    "explanation": "explainability",
    "explanations": "explainability",
    "results": "explainability",
}

DEFAULT_STAGE_CONFIGS: dict[str, dict[str, Any]] = {
    "analysis": {},
    "preprocessing": {
        "missing_value_strategy": "auto",
        "protect_rows_from_drop": False,
        "force_keep_columns": [],
        "force_drop_columns": [],
        "encoding_strategy_overrides": {},
        "scaler": "auto",
        "rare_category_grouping": True,
    },
    "feature_engineering": {
        "include_features": [],
        "exclude_features": [],
        "force_keep_engineered_features": [],
        "force_drop_engineered_features": [],
        "importance_threshold": None,
        "correlation_threshold": None,
        "use_interactions": True,
        "use_pca": False,
        "n_pca_components": None,
    },
    "training": {
        "preferred_model_family": None,
        "force_model_name": None,
        "reduce_complexity": False,
        "regularization_strength": "normal",
        "cv_folds": None,
        "enable_class_weights": False,
        "metric_priority": None,
        "retune_hyperparameters": True,
    },
    "evaluation": {
        "primary_metric": None,
        "deployment_threshold": None,
        "rerun_baseline_comparison": True,
    },
    "explainability": {
        "source": "auto",
        "prefer_shap": False,
        "fallback_importance": True,
    },
}


def normalize_stage_name(stage: str) -> str:
    """Normalize a user-facing or internal stage name into a canonical stage."""
    normalized = str(stage or "").strip().lower().replace("-", "_").replace(" ", "_")
    return STAGE_NAME_ALIASES.get(normalized, normalized)


def canonical_downstream_stages(stage: str) -> list[str]:
    """Return canonical stages from the selected start stage onward."""
    canonical_stage = normalize_stage_name(stage)
    if canonical_stage not in CANONICAL_STAGE_ORDER:
        raise ValueError(f"Unsupported canonical stage: {stage}")
    start_index = CANONICAL_STAGE_ORDER.index(canonical_stage)
    return CANONICAL_STAGE_ORDER[start_index:]


def concrete_stages_from_canonical(stage: str) -> list[str]:
    """Return concrete runtime stages from the selected canonical start stage onward."""
    canonical_stage = normalize_stage_name(stage)
    concrete_start = CANONICAL_TO_CONCRETE_STAGE.get(canonical_stage, canonical_stage)
    if concrete_start not in CONCRETE_STAGE_ORDER:
        raise ValueError(f"Unsupported concrete stage: {stage}")
    start_index = CONCRETE_STAGE_ORDER.index(concrete_start)
    return CONCRETE_STAGE_ORDER[start_index:]


@dataclass
class PipelineRunRecord:
    """Persisted summary of a pipeline run or revision."""

    run_id: str
    parent_run_id: Optional[str]
    dataset: dict[str, Any]
    target_column: str
    task_type: str
    stage_configs: dict[str, dict[str, Any]]
    stage_outputs: dict[str, Any]
    metrics: dict[str, Any]
    revision_reason: Optional[str]
    changed_stages: list[str]
    changed_configs: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return asdict(self)


class PipelineState:
    """Mutable runtime state plus revision history for the current dataset."""

    def __init__(self) -> None:
        self.dataset: Optional[pd.DataFrame] = None
        self.dataset_path: Optional[str] = None
        self.dataset_filename: Optional[str] = None
        self.target_column: Optional[str] = None
        self.task_type: str = "classification"
        self.pipeline_id: Optional[str] = None
        self.pipeline_config: dict[str, Any] = {
            "task_type": "classification",
            "test_size": 0.2,
            "random_state": 42,
        }
        self.stage_results: dict[str, Any] = {}
        self.stage_statuses: dict[str, str] = {
            "analysis": "waiting",
            "preprocessing": "waiting",
            "features": "waiting",
            "model_selection": "waiting",
            "training": "waiting",
            "loss": "waiting",
            "evaluation": "waiting",
            "results": "waiting",
        }
        self.stage_logs: dict[str, list[str]] = {
            stage: [] for stage in self.stage_statuses
        }
        self.stage_configs: dict[str, dict[str, Any]] = deepcopy(DEFAULT_STAGE_CONFIGS)
        self.revision_history: list[PipelineRunRecord] = []
        self.pending_revision_plan: Optional[dict[str, Any]] = None

    @property
    def current_run_id(self) -> Optional[str]:
        """Return the current run identifier if any."""
        if not self.revision_history:
            return None
        return self.revision_history[-1].run_id

    def reset_for_dataset(
        self,
        *,
        df: pd.DataFrame,
        dataset_path: str,
        dataset_filename: str,
        pipeline_id: str,
    ) -> None:
        """Reset runtime state for a newly uploaded dataset."""
        self.dataset = df
        self.dataset_path = dataset_path
        self.dataset_filename = dataset_filename
        self.pipeline_id = pipeline_id
        self.target_column = None
        self.task_type = "classification"
        self.pipeline_config = {
            "task_type": "classification",
            "test_size": 0.2,
            "random_state": 42,
        }
        self.stage_results = {}
        self.stage_statuses = {stage: "waiting" for stage in self.stage_statuses}
        self.stage_logs = {stage: [] for stage in self.stage_logs}
        self.stage_configs = deepcopy(DEFAULT_STAGE_CONFIGS)
        self.revision_history = []
        self.pending_revision_plan = None

    def update_pipeline_config(self, config: dict[str, Any]) -> None:
        """Store the latest top-level pipeline config."""
        self.pipeline_config.update(config)
        self.task_type = str(config.get("task_type") or self.task_type)

    def dataset_info(self) -> dict[str, Any]:
        """Return lightweight dataset metadata."""
        df = self.dataset
        return {
            "filename": self.dataset_filename,
            "path": self.dataset_path,
            "rows": int(len(df)) if df is not None else 0,
            "columns": int(len(df.columns)) if df is not None else 0,
            "column_names": [str(column) for column in df.columns] if df is not None else [],
        }

    def known_features(self) -> list[str]:
        """Return known feature names from the dataset and current stage outputs."""
        names: list[str] = []
        if self.dataset is not None:
            names.extend(str(column) for column in self.dataset.columns if column != self.target_column)

        features_result = self.stage_results.get("features", {}) or {}
        for key in ("selected_features", "generated_features", "dropped_columns"):
            values = features_result.get(key, [])
            if isinstance(values, list):
                names.extend(str(item) for item in values)

        preprocessing_result = self.stage_results.get("preprocessing", {}) or {}
        dropped_columns = preprocessing_result.get("dropped_columns", [])
        if isinstance(dropped_columns, list):
            for item in dropped_columns:
                if isinstance(item, dict) and item.get("column"):
                    names.append(str(item["column"]))

        seen: set[str] = set()
        ordered: list[str] = []
        for name in names:
            lowered = name.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            ordered.append(name)
        return ordered

    def structured_stage_outputs(self) -> dict[str, Any]:
        """Return public stage outputs keyed by canonical stage names."""
        return {
            "analysis": public_value(self.stage_results.get("analysis", {})),
            "preprocessing": public_value(self.stage_results.get("preprocessing", {})),
            "feature_engineering": public_value(self.stage_results.get("features", {})),
            "training": public_value({
                "model_selection": self.stage_results.get("model_selection", {}),
                "training": self.stage_results.get("training", {}),
            }),
            "evaluation": public_value(self.stage_results.get("evaluation", {})),
            "explainability": public_value({
                "results": self.stage_results.get("results", {}),
                "explanation": self.stage_results.get("explanation", {}),
            }),
        }

    def metrics_summary(self) -> dict[str, Any]:
        """Return a compact metrics view for planning and comparisons."""
        training = self.stage_results.get("training", {}) or {}
        evaluation = self.stage_results.get("evaluation", {}) or {}
        return public_value(
            {
                "task_type": evaluation.get("task_type", self.task_type),
                "model_name": training.get("model_name"),
                "best_score": training.get("best_score"),
                "train_score": training.get("train_score"),
                "test_score": training.get("test_score"),
                "accuracy": evaluation.get("accuracy"),
                "precision": evaluation.get("precision"),
                "recall": evaluation.get("recall"),
                "f1": evaluation.get("f1"),
                "roc_auc": evaluation.get("roc_auc"),
                "r2": evaluation.get("r2"),
                "mae": evaluation.get("mae"),
                "mse": evaluation.get("mse"),
                "rmse": evaluation.get("rmse"),
                "deployment_decision": evaluation.get("deployment_decision"),
                "selected_features": training.get("selected_features")
                or self.stage_results.get("features", {}).get("selected_features", []),
                "hyperparameters": training.get("hyperparameters")
                or training.get("best_params"),
            }
        )

    def current_structured_state(self) -> dict[str, Any]:
        """Return the current pipeline state in a structured form."""
        return {
            "run_id": self.current_run_id,
            "parent_run_id": self.revision_history[-1].parent_run_id if self.revision_history else None,
            "dataset": self.dataset_info(),
            "target_column": self.target_column,
            "task_type": self.task_type,
            "stage_configs": deepcopy(self.stage_configs),
            "stage_outputs": self.structured_stage_outputs(),
            "metrics": self.metrics_summary(),
            "revision_reason": self.revision_history[-1].revision_reason if self.revision_history else None,
            "changed_stages": self.revision_history[-1].changed_stages if self.revision_history else [],
            "created_at": self.revision_history[-1].created_at if self.revision_history else None,
        }

    def build_run_record(
        self,
        *,
        revision_reason: Optional[str],
        changed_stages: list[str],
        changed_configs: Optional[dict[str, Any]] = None,
        parent_run_id: Optional[str] = None,
    ) -> PipelineRunRecord:
        """Create a new run record from the current runtime state."""
        if not self.target_column:
            raise ValueError("Target column must be set before recording a run")

        return PipelineRunRecord(
            run_id=str(uuid4()),
            parent_run_id=parent_run_id,
            dataset=self.dataset_info(),
            target_column=self.target_column,
            task_type=self.task_type,
            stage_configs=deepcopy(self.stage_configs),
            stage_outputs=self.structured_stage_outputs(),
            metrics=self.metrics_summary(),
            revision_reason=revision_reason,
            changed_stages=list(changed_stages),
            changed_configs=public_value(changed_configs or {}),
        )
