"""Deterministic revision action registry for pipeline-safe edits."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.diff_utils import diff_mapping
from core.pipeline_state import normalize_stage_name


class ActionValidationError(ValueError):
    """Raised when a proposed revision action is invalid."""


class ControlledActionRegistry:
    """Apply validated revision actions to stage configs."""

    def apply_actions(
        self,
        *,
        current_state: dict[str, Any],
        stage_configs: dict[str, dict[str, Any]],
        actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply actions and return updated configs plus diffs."""
        updated = deepcopy(stage_configs)
        known_features = {
            str(name).lower(): str(name)
            for name in current_state.get("known_features", [])
        }

        for action in actions:
            stage = normalize_stage_name(str(action.get("stage") or "").strip())
            action_type = str(action.get("action_type") or "").strip()
            params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}
            if stage not in updated:
                raise ActionValidationError(f"Unsupported stage for revision: {stage}")
            handler = getattr(self, f"_handle_{action_type}", None)
            if handler is None:
                raise ActionValidationError(f"Unsupported revision action: {action_type}")
            handler(updated, params, known_features)

        config_diffs = {
            stage: diff_mapping(stage_configs.get(stage, {}), updated.get(stage, {}))
            for stage in updated
        }
        changed_stages = [
            stage for stage, diff in config_diffs.items()
            if diff not in ({}, None)
        ]
        return {
            "stage_configs": updated,
            "changed_stages": changed_stages,
            "config_diffs": {
                stage: diff for stage, diff in config_diffs.items()
                if diff not in ({}, None)
            },
        }

    def _resolve_feature_name(
        self,
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> str:
        raw = str(params.get("feature_name") or "").strip()
        if not raw:
            raise ActionValidationError("Feature-based revisions require a feature name")
        return known_features.get(raw.lower(), raw)

    def _toggle_unique(self, items: list[Any], value: Any, enabled: bool) -> list[Any]:
        next_items = [item for item in items if item != value]
        if enabled:
            next_items.append(value)
        return next_items

    def _handle_change_missing_value_strategy(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        strategy = str(params.get("strategy") or "auto")
        allowed = {"auto", "impute", "drop_rows", "drop_columns", "mixed"}
        if strategy not in allowed:
            raise ActionValidationError(f"Unsupported missing-value strategy: {strategy}")
        configs["preprocessing"]["missing_value_strategy"] = strategy
        configs["preprocessing"]["protect_rows_from_drop"] = strategy == "impute"

    def _handle_protect_rows_from_drop(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        enabled = bool(params.get("enabled", True))
        configs["preprocessing"]["protect_rows_from_drop"] = enabled
        if enabled:
            configs["preprocessing"]["missing_value_strategy"] = "impute"

    def _handle_drop_column(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        feature_name = self._resolve_feature_name(params, known_features)
        configs["preprocessing"]["force_drop_columns"] = self._toggle_unique(
            list(configs["preprocessing"].get("force_drop_columns", [])),
            feature_name,
            True,
        )
        configs["preprocessing"]["force_keep_columns"] = self._toggle_unique(
            list(configs["preprocessing"].get("force_keep_columns", [])),
            feature_name,
            False,
        )

    def _handle_keep_column(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        feature_name = self._resolve_feature_name(params, known_features)
        configs["preprocessing"]["force_keep_columns"] = self._toggle_unique(
            list(configs["preprocessing"].get("force_keep_columns", [])),
            feature_name,
            True,
        )
        configs["preprocessing"]["force_drop_columns"] = self._toggle_unique(
            list(configs["preprocessing"].get("force_drop_columns", [])),
            feature_name,
            False,
        )

    def _handle_change_encoding_strategy(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        feature_name = self._resolve_feature_name(params, known_features)
        strategy = str(params.get("strategy") or "").strip()
        if strategy not in {"onehot", "frequency", "binary"}:
            raise ActionValidationError(f"Unsupported encoding strategy: {strategy}")
        overrides = dict(configs["preprocessing"].get("encoding_strategy_overrides", {}))
        overrides[feature_name] = strategy
        configs["preprocessing"]["encoding_strategy_overrides"] = overrides

    def _handle_change_scaler(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        scaler = str(params.get("scaler") or "auto")
        if scaler not in {"auto", "StandardScaler", "RobustScaler", "None"}:
            raise ActionValidationError(f"Unsupported scaler: {scaler}")
        configs["preprocessing"]["scaler"] = scaler

    def _handle_toggle_rare_category_grouping(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        configs["preprocessing"]["rare_category_grouping"] = bool(params.get("enabled", True))

    def _handle_include_feature(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        feature_name = self._resolve_feature_name(params, known_features)
        configs["feature_engineering"]["include_features"] = self._toggle_unique(
            list(configs["feature_engineering"].get("include_features", [])),
            feature_name,
            True,
        )
        configs["feature_engineering"]["exclude_features"] = self._toggle_unique(
            list(configs["feature_engineering"].get("exclude_features", [])),
            feature_name,
            False,
        )

    def _handle_exclude_feature(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        feature_name = self._resolve_feature_name(params, known_features)
        configs["feature_engineering"]["exclude_features"] = self._toggle_unique(
            list(configs["feature_engineering"].get("exclude_features", [])),
            feature_name,
            True,
        )
        configs["feature_engineering"]["include_features"] = self._toggle_unique(
            list(configs["feature_engineering"].get("include_features", [])),
            feature_name,
            False,
        )

    def _handle_change_importance_threshold(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        configs["feature_engineering"]["importance_threshold"] = float(params.get("value"))

    def _handle_change_correlation_threshold(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        configs["feature_engineering"]["correlation_threshold"] = float(params.get("value"))

    def _handle_toggle_interactions(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        configs["feature_engineering"]["use_interactions"] = bool(params.get("enabled", True))

    def _handle_toggle_pca(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        configs["feature_engineering"]["use_pca"] = bool(params.get("enabled", True))
        if "n_components" in params and params.get("n_components") is not None:
            configs["feature_engineering"]["n_pca_components"] = int(params["n_components"])

    def _handle_force_keep_engineered_feature(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        feature_name = self._resolve_feature_name(params, known_features)
        configs["feature_engineering"]["force_keep_engineered_features"] = self._toggle_unique(
            list(configs["feature_engineering"].get("force_keep_engineered_features", [])),
            feature_name,
            True,
        )

    def _handle_force_drop_engineered_feature(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        feature_name = self._resolve_feature_name(params, known_features)
        configs["feature_engineering"]["force_drop_engineered_features"] = self._toggle_unique(
            list(configs["feature_engineering"].get("force_drop_engineered_features", [])),
            feature_name,
            True,
        )

    def _handle_switch_model_family(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        if params.get("model_name"):
            configs["training"]["force_model_name"] = str(params["model_name"])
            configs["training"]["preferred_model_family"] = None
            return

        family = str(params.get("model_family") or "").strip()
        if family not in {"linear", "tree_ensemble", "boosted_trees", "kernel"}:
            raise ActionValidationError(f"Unsupported model family: {family}")
        configs["training"]["preferred_model_family"] = family
        configs["training"]["force_model_name"] = None

    def _handle_reduce_model_complexity(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        configs["training"]["reduce_complexity"] = True

    def _handle_increase_regularization(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        level = str(configs["training"].get("regularization_strength") or "normal")
        next_level = {
            "normal": "high",
            "high": "very_high",
            "very_high": "very_high",
        }.get(level, "high")
        configs["training"]["regularization_strength"] = next_level

    def _handle_increase_cv_folds(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        current = configs["training"].get("cv_folds") or 5
        target = int(params.get("value") or current + 2)
        configs["training"]["cv_folds"] = max(int(current), target)

    def _handle_enable_class_weights(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        configs["training"]["enable_class_weights"] = bool(params.get("enabled", True))

    def _handle_change_metric_priority(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        metric = str(params.get("metric") or "").strip()
        configs["training"]["metric_priority"] = metric or None
        configs["evaluation"]["primary_metric"] = metric or None

    def _handle_retune_hyperparameters(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        configs["training"]["retune_hyperparameters"] = bool(params.get("enabled", True))

    def _handle_change_primary_metric(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        configs["evaluation"]["primary_metric"] = str(params.get("metric") or "").strip() or None

    def _handle_change_deployment_threshold(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        configs["evaluation"]["deployment_threshold"] = float(params.get("value"))

    def _handle_rerun_baseline_comparison(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        configs["evaluation"]["rerun_baseline_comparison"] = bool(params.get("enabled", True))

    def _handle_change_explainability_source(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        configs["explainability"]["source"] = str(params.get("source") or "auto")

    def _handle_prefer_shap(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        configs["explainability"]["prefer_shap"] = bool(params.get("enabled", True))

    def _handle_use_fallback_importance(
        self,
        configs: dict[str, dict[str, Any]],
        params: dict[str, Any],
        known_features: dict[str, str],
    ) -> None:
        configs["explainability"]["fallback_importance"] = bool(params.get("enabled", True))
