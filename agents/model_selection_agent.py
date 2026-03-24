"""Model Selection Agent for AutoML Pipeline.

This agent recommends machine learning models based on:
- Data characteristics
- Task type (classification/regression)
- Dataset size
- Feature types
"""

import logging
from typing import Any, Optional

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

from agents.base_agent import BaseAgent
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class ModelSelectionAgent(BaseAgent):
    """Agent for selecting appropriate ML models based on data characteristics.

    This agent recommends models based on:
    - Dataset size
    - Feature types
    - Task type (classification/regression)
    - Class imbalance
    """

    def __init__(self) -> None:
        """Initialize the ModelSelectionAgent."""
        super().__init__("ModelSelection")

    async def execute(
        self,
        df: pd.DataFrame,
        features: dict[str, Any],
        target_column: str,
        task_type: str = "classification",
        analysis: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Select appropriate models for the dataset.

        Args:
            df: The input DataFrame.
            features: Feature engineering results from FeatureEngineeringAgent.
            target_column: The target column name.
            task_type: Type of task - "classification" or "regression".
            analysis: Analysis results from DataAnalyzerAgent (optional).

        Returns:
            Dictionary containing model selection results including:
            - top_candidates (up to 3 concrete algorithms with equal weighting)
            - selection_reasoning
            - candidate-level fixed params and search spaces
        """
        try:
            logger.info(f"Selecting models for {task_type} task")

            # Analyze data characteristics
            n_samples = len(df)
            n_features = features.get("final_feature_count", df.shape[1] - 1)

            # Extract analysis metrics if available
            analysis_metrics = self._extract_analysis_metrics(analysis, n_samples, n_features) if analysis else {}

            # Check class distribution
            y = df[target_column]
            if task_type == "classification":
                class_counts = y.value_counts()
                min_class_ratio = class_counts.min() / class_counts.max() if len(class_counts) > 0 else 1.0
            else:
                min_class_ratio = 1.0

            # Select candidate models based on characteristics
            candidate_models = self._select_candidates(
                n_samples=n_samples,
                n_features=n_features,
                task_type=task_type,
                class_balance=min_class_ratio,
                analysis=analysis_metrics,
            )

            limited_candidates = candidate_models[:3]

            selection_reasoning = self._generate_reasoning(
                n_samples=n_samples,
                n_features=n_features,
                task_type=task_type,
                class_balance=min_class_ratio,
                candidate_models=limited_candidates,
                analysis=analysis_metrics if analysis else {},
            )

            total_candidates = max(1, len(limited_candidates))

            top_candidates = [
                self._build_candidate_payload(
                    model_name=model_name,
                    priority=index + 1,
                    total_candidates=total_candidates,
                    n_samples=n_samples,
                    task_type=task_type,
                    class_balance=min_class_ratio,
                    analysis=analysis_metrics if analysis else {},
                )
                for index, model_name in enumerate(limited_candidates)
            ]

            if not top_candidates:
                top_candidates = [
                    self._build_candidate_payload(
                        model_name="RandomForest",
                        priority=1,
                        total_candidates=1,
                        n_samples=n_samples,
                        task_type=task_type,
                        class_balance=min_class_ratio,
                        analysis=analysis_metrics if analysis else {},
                    )
                ]

            llm_selection = self._generate_llm_selection(
                n_samples=n_samples,
                n_features=n_features,
                task_type=task_type,
                class_balance=min_class_ratio,
                candidate_models=limited_candidates,
                default_candidates=top_candidates,
                default_reasoning=selection_reasoning,
                target_column=target_column,
                selected_features=features.get("selected_features", []),
                analysis=analysis_metrics if analysis else {},
            )

            llm_returned = bool(llm_selection)

            if not llm_selection:
                llm_selection = {
                    "top_candidates": top_candidates,
                    "selection_reasoning": selection_reasoning,
                }

            merged_candidates = self._merge_candidate_selection(
                default_candidates=top_candidates,
                llm_candidates=llm_selection.get("top_candidates"),
                candidate_models=limited_candidates,
                n_samples=n_samples,
                task_type=task_type,
                class_balance=min_class_ratio,
                analysis=analysis_metrics if analysis else {},
            )
            selection_reasoning = str(llm_selection.get("selection_reasoning") or selection_reasoning)

            llm_summary = self._build_selection_summary(
                n_samples=n_samples,
                n_features=n_features,
                task_type=task_type,
                class_balance=min_class_ratio,
                candidate_models=[candidate["model_name"] for candidate in merged_candidates],
                selected_model=merged_candidates[0]["model_name"] if merged_candidates else "RandomForest",
                top_candidates=merged_candidates,
                reasoning=selection_reasoning,
                target_column=target_column,
                selected_features=features.get("selected_features", []),
                analysis=analysis_metrics if analysis else {},
            )

            result = {
                "top_candidates": merged_candidates,
                "selection_reasoning": selection_reasoning,
                "task_type": task_type,
                "n_samples": n_samples,
                "n_features": n_features,
                "selection_source": "llm" if llm_returned else "heuristic",
                "selected_features": features.get("selected_features", []),
                "feature_scores": features.get("feature_scores", {}),
                "applied_transformations": features.get("applied_transformations", []),
                "_engineered_df": features.get("_engineered_df"),
                "llm_returned": llm_returned,
                "llm_summary": llm_summary,
                "analysis_signals": analysis_metrics if analysis else {},
                "class_balance": min_class_ratio,
            }

            selected_name = merged_candidates[0]["model_name"] if merged_candidates else "RandomForest"
            logger.info(f"Model selection complete: candidate set includes {selected_name}")
            return result

        except Exception as e:
            logger.exception(f"Error in model selection: {e}")
            raise AgentExecutionError(
                f"Model selection failed: {str(e)}",
                agent_name=self.name,
                details={"error": str(e)},
            ) from e

    def _extract_analysis_metrics(
        self,
        analysis: dict[str, Any],
        n_samples: int,
        n_features: int,
    ) -> dict[str, Any]:
        """Extract relevant metrics from DataAnalyzer output."""
        metrics = {}

        # Missing values
        missing_values = analysis.get("missing_values", {})
        if missing_values:
            total_missing = sum(missing_values.values())
            avg_missing = total_missing / len(missing_values) if missing_values else 0
            metrics["avg_missing_pct"] = avg_missing
            metrics["high_missing_cols"] = len(analysis.get("high_missing_columns", []))

        # Correlations
        high_corr_pairs = analysis.get("high_correlation_pairs", [])
        metrics["high_correlation_count"] = len(high_corr_pairs)
        metrics["has_high_correlations"] = len(high_corr_pairs) > 3

        # Outliers
        outliers = analysis.get("outliers", {})
        metrics["outlier_columns"] = len(outliers)
        metrics["has_outliers"] = len(outliers) > 0

        # Column types
        numeric_cols = analysis.get("numeric_columns", [])
        categorical_cols = analysis.get("categorical_columns", [])
        metrics["numeric_count"] = len(numeric_cols)
        metrics["categorical_count"] = len(categorical_cols)

        # Risk level
        metrics["risk_level"] = analysis.get("risk_level", "low")

        # Calculate derived metrics
        total_cols = metrics.get("numeric_count", 0) + metrics.get("categorical_count", 0)
        if total_cols > 0:
            metrics["dimensionality_ratio"] = n_features / total_cols
        else:
            metrics["dimensionality_ratio"] = 0

        return metrics

    def _select_candidates(
        self,
        n_samples: int,
        n_features: int,
        task_type: str,
        class_balance: float,
        analysis: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        """Select candidate models based on data characteristics."""
        candidates = []

        # Initialize analysis flags
        has_high_correlations = False
        has_high_missing = False
        has_outliers = False
        is_high_dimensional = False
        dimensionality_ratio = 0

        if analysis:
            has_high_correlations = analysis.get("has_high_correlations", False)
            has_high_missing = analysis.get("high_missing_cols", 0) > 2
            has_outliers = analysis.get("has_outliers", False)
            dimensionality_ratio = analysis.get("dimensionality_ratio", 0)
            # High-dimensional: features > 50% of samples
            is_high_dimensional = n_features > n_samples * 0.5

        if task_type == "classification":
            if n_samples < 1000:
                candidates.extend(["LogisticRegression", "SVM", "RandomForest"])
            elif n_samples < 10000:
                candidates.extend(["RandomForest", "GradientBoosting", "XGBoost", "LightGBM"])
            else:
                candidates.extend(["RandomForest", "GradientBoosting", "XGBoost", "LightGBM"])

            # Adjust for class imbalance
            if class_balance < 0.1:
                candidates.extend(["XGBoost", "LightGBM", "RandomForest"])

            # Adjust for high-dimensional sparse data (features >> samples)
            if is_high_dimensional:
                # Use regularized linear models
                candidates.extend(["LogisticRegression"])
                # Avoid SVM with RBF kernel on high-dim data
                if "SVM" in candidates:
                    candidates.remove("SVM")

            # Adjust for high correlations - avoid models sensitive to multicollinearity
            if has_high_correlations:
                # Tree-based models are more robust
                candidates.extend(["RandomForest", "GradientBoosting", "LightGBM"])

            # Adjust for high missing values - prefer models robust to missing data
            if has_high_missing:
                candidates.extend(["RandomForest", "GradientBoosting", "LightGBM"])

            # Adjust for outliers - avoid SVM
            if has_outliers:
                if "SVM" in candidates:
                    candidates.remove("SVM")
                candidates.extend(["RandomForest", "GradientBoosting"])
        else:
            if n_samples < 1000:
                candidates.extend(["Ridge", "SVR", "RandomForest"])
            elif n_samples < 10000:
                candidates.extend(["RandomForest", "GradientBoosting", "XGBoost", "Ridge"])
            else:
                candidates.extend(["RandomForest", "GradientBoosting", "XGBoost", "LightGBM"])

            # Adjust for high-dimensional sparse data
            if is_high_dimensional:
                candidates.extend(["Ridge"])

            # Adjust for high correlations
            if has_high_correlations:
                candidates.extend(["Ridge", "LightGBM"])

            # Adjust for outliers
            if has_outliers:
                if "SVR" in candidates:
                    candidates.remove("SVR")
                candidates.extend(["RandomForest", "GradientBoosting", "LightGBM"])

        return list(dict.fromkeys(candidates))

    def _build_candidate_payload(
        self,
        model_name: str,
        priority: int,
        total_candidates: int,
        n_samples: int,
        task_type: str,
        class_balance: float,
        analysis: Optional[dict[str, Any]],
        fixed_params_override: Optional[dict[str, Any]] = None,
        search_space_override: Optional[dict[str, Any]] = None,
        reasoning_override: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build a normalized candidate payload for training handoff."""
        fixed_params = self._get_default_hyperparameters(model_name, n_samples, task_type)
        if isinstance(fixed_params_override, dict):
            fixed_params.update(self._sanitize_hyperparameters(model_name, fixed_params_override, task_type))

        candidate_search_space = self._get_search_space_for_model(model_name)
        if isinstance(search_space_override, dict) and search_space_override:
            candidate_search_space = self._sanitize_search_space(search_space_override)

        return {
            "priority": priority,
            "candidate_weight": round(1.0 / max(1, total_candidates), 4),
            "model_name": model_name,
            "model_family": self._resolve_model_family(model_name),
            "reasoning": reasoning_override or self._build_candidate_reasoning(
                model_name=model_name,
                class_balance=class_balance,
                task_type=task_type,
                analysis=analysis,
            ),
            "fixed_params": fixed_params,
            "search_space": candidate_search_space,
        }

    def _get_default_hyperparameters(
        self,
        model_name: str,
        n_samples: int,
        task_type: str,
    ) -> dict[str, Any]:
        """Get default hyperparameters for a model."""
        base_params = {"random_state": 42}

        if model_name == "RandomForest":
            n_estimators = min(100, max(10, n_samples // 100))
            return {
                **base_params,
                "n_estimators": n_estimators,
                "max_depth": 10,
                "min_samples_split": 5,
            }
        elif model_name == "GradientBoosting":
            return {
                **base_params,
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
            }
        elif model_name == "XGBoost":
            return {
                **base_params,
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "eval_metric": "logloss",
            }
        elif model_name == "LightGBM":
            return {
                **base_params,
                "n_estimators": 100,
                "max_depth": 8,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "verbose": -1,
            }
        elif model_name == "LogisticRegression":
            return {
                **base_params,
                "max_iter": 1000,
                "C": 1.0,
            }
        elif model_name == "Ridge":
            return {"alpha": 1.0, **base_params}
        elif model_name == "SVR":
            return {"kernel": "rbf", "C": 1.0}
        else:
            return base_params

    def _generate_reasoning(
        self,
        n_samples: int,
        n_features: int,
        task_type: str,
        class_balance: float,
        candidate_models: list[str],
        analysis: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate reasoning for model selection."""
        reasoning_parts = []

        reasoning_parts.append(f"Dataset has {n_samples} samples and {n_features} features.")

        if n_samples < 1000:
            reasoning_parts.append("Small dataset - simpler models are less likely to overfit.")
        elif n_samples > 10000:
            reasoning_parts.append("Large dataset - can leverage more complex models.")

        if analysis:
            if analysis.get("has_high_correlations"):
                reasoning_parts.append("High feature correlations detected - using models robust to multicollinearity.")
            if analysis.get("has_outliers"):
                reasoning_parts.append("Outliers present in data - tree-based models recommended for robustness.")
            if analysis.get("high_missing_cols", 0) > 2:
                reasoning_parts.append(f"High missing values in {analysis.get('high_missing_cols')} columns.")
            if analysis.get("dimensionality_ratio", 0) > 0.5:
                reasoning_parts.append("High-dimensional data - considering regularized models.")

        if class_balance < 0.1 and task_type == "classification":
            reasoning_parts.append("Class imbalance detected - ensemble methods recommended.")

        if candidate_models:
            reasoning_parts.append(
                "Selected a balanced candidate set with equal priority weighting: "
                + ", ".join(candidate_models)
                + "."
            )

        return " ".join(reasoning_parts)

    def _generate_llm_selection(
        self,
        *,
        n_samples: int,
        n_features: int,
        task_type: str,
        class_balance: float,
        candidate_models: list[str],
        default_candidates: list[dict[str, Any]],
        default_reasoning: str,
        target_column: str,
        selected_features: list[str],
        analysis: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Use an LLM to refine model choice within deterministic guardrails."""
        payload = {
            "task_type": task_type,
            "target_column": target_column,
            "n_samples": n_samples,
            "n_features": n_features,
            "class_balance": class_balance,
            "candidate_models": candidate_models,
            "selected_features_preview": selected_features[:15],
            "default_candidates": default_candidates,
            "default_reasoning": default_reasoning,
        }

        # Add analysis data to payload for smarter recommendations
        if analysis:
            payload["data_analysis"] = {
                "high_correlation_count": analysis.get("high_correlation_count", 0),
                "has_high_correlations": analysis.get("has_high_correlations", False),
                "high_missing_cols": analysis.get("high_missing_cols", 0),
                "has_outliers": analysis.get("has_outliers", False),
                "is_high_dimensional": analysis.get("dimensionality_ratio", 0) > 0.5 if analysis.get("dimensionality_ratio") else False,
                "risk_level": analysis.get("risk_level", "low"),
                "numeric_count": analysis.get("numeric_count", 0),
                "categorical_count": analysis.get("categorical_count", 0),
            }

        response = self._generate_llm_json(
            system_prompt=(
                "You are an AutoML model-selection assistant. "
                "Return JSON with keys 'top_candidates' and 'selection_reasoning'. "
                "top_candidates must be an ordered list with up to 3 items where each item has: "
                "'model_name', 'reasoning', 'fixed_params', and optional 'search_space'. "
                "Choose models ONLY from candidate_models. Treat all returned candidates as equally weighted options "
                "and avoid language that labels one model as the primary winner. "
                "Use constructor-safe hyperparameters. "
                "Consider data characteristics: high correlations favor tree models, "
                "high dimensionality favors regularized linear models, "
                "outliers favor tree-based models over SVM."
            ),
            user_prompt=f"Model selection context:\n{self._safe_json(payload)}",
            temperature=0.1,
            max_tokens=2000,
        )

        if not response:
            return None

        raw_top_candidates = response.get("top_candidates")
        if not isinstance(raw_top_candidates, list):
            return None

        llm_candidates: list[dict[str, Any]] = []
        for index, item in enumerate(raw_top_candidates[:3]):
            if not isinstance(item, dict):
                continue
            proposed_model = str(item.get("model_name") or "").strip()
            if proposed_model not in candidate_models:
                continue
            llm_candidates.append(
                {
                    "priority": index + 1,
                    "model_name": proposed_model,
                    "reasoning": str(item.get("reasoning") or "").strip(),
                    "fixed_params": item.get("fixed_params", {}),
                    "search_space": item.get("search_space", {}),
                }
            )

        if not llm_candidates:
            return None

        reasoning = str(response.get("selection_reasoning") or default_reasoning)
        return {
            "top_candidates": llm_candidates,
            "selection_reasoning": reasoning,
        }

    def _build_selection_summary(
        self,
        *,
        n_samples: int,
        n_features: int,
        task_type: str,
        class_balance: float,
        candidate_models: list[str],
        selected_model: str,
        top_candidates: list[dict[str, Any]],
        reasoning: str,
        target_column: str,
        selected_features: list[str],
        analysis: Optional[dict[str, Any]] = None,
    ) -> str:
        """Build a short deterministic summary without a second LLM call."""
        candidate_label = ", ".join(candidate_models[:3]) if candidate_models else selected_model
        feature_count = len(selected_features)
        analysis_notes: list[str] = []

        if analysis:
            if analysis.get("has_high_correlations"):
                analysis_notes.append("high feature correlation")
            if analysis.get("has_outliers"):
                analysis_notes.append("outlier sensitivity")
            if analysis.get("high_missing_cols", 0) > 0:
                analysis_notes.append("missing-value risk")
            if analysis.get("dimensionality_ratio", 0) > 0.5:
                analysis_notes.append("high dimensionality")

        summary_parts = [
            f"Prepared a candidate set for the {task_type} target `{target_column}` using {n_samples} rows and {n_features} features.",
            f"Current candidates: {candidate_label}.",
        ]

        if feature_count > 0:
            summary_parts.append(f"The selection considered {feature_count} engineered/selected features from upstream stages.")

        if class_balance < 0.999 and task_type == "classification":
            summary_parts.append(f"Class balance was reviewed at roughly {(class_balance * 100):.1f}% minority-to-majority ratio.")

        if analysis_notes:
            summary_parts.append(f"Data signals considered: {', '.join(analysis_notes)}.")

        cleaned_reasoning = reasoning.strip()
        if cleaned_reasoning:
            summary_parts.append(cleaned_reasoning)

        return " ".join(part for part in summary_parts if part).strip()

    def _merge_candidate_selection(
        self,
        default_candidates: list[dict[str, Any]],
        llm_candidates: Any,
        candidate_models: list[str],
        n_samples: int,
        task_type: str,
        class_balance: float,
        analysis: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge LLM candidate details with deterministic defaults."""
        if not isinstance(llm_candidates, list):
            return default_candidates

        merged: list[dict[str, Any]] = []
        for index, item in enumerate(llm_candidates[:3]):
            if not isinstance(item, dict):
                continue
            model_name = str(item.get("model_name") or "").strip()
            if model_name not in candidate_models:
                continue

            merged.append(
                self._build_candidate_payload(
                    model_name=model_name,
                    priority=index + 1,
                    total_candidates=max(1, len(llm_candidates[:3])),
                    n_samples=n_samples,
                    task_type=task_type,
                    class_balance=class_balance,
                    analysis=analysis,
                    fixed_params_override=item.get("fixed_params") if isinstance(item.get("fixed_params"), dict) else None,
                    search_space_override=item.get("search_space") if isinstance(item.get("search_space"), dict) else None,
                    reasoning_override=str(item.get("reasoning") or "").strip() or None,
                )
            )

        if not merged:
            return default_candidates

        selected_names = {candidate["model_name"] for candidate in merged}
        fallback_pool = [name for name in candidate_models if name not in selected_names]
        while len(merged) < 3 and fallback_pool:
            model_name = fallback_pool.pop(0)
            merged.append(
                self._build_candidate_payload(
                    model_name=model_name,
                    priority=len(merged) + 1,
                    total_candidates=3,
                    n_samples=n_samples,
                    task_type=task_type,
                    class_balance=class_balance,
                    analysis=analysis,
                )
            )

        total_candidates = max(1, len(merged))
        for index, candidate in enumerate(merged, start=1):
            candidate["priority"] = index
            candidate["candidate_weight"] = round(1.0 / total_candidates, 4)
        return merged

    def _resolve_model_family(self, model_name: str) -> str:
        """Map model names to canonical family labels."""
        normalized = model_name.lower().replace("-", "").replace("_", "")
        if "randomforest" in normalized:
            return "tree_ensemble"
        if "gradientboosting" in normalized or "xgboost" in normalized or "lightgbm" in normalized:
            return "boosted_trees"
        if "logistic" in normalized or "ridge" in normalized:
            return "linear"
        if "svm" in normalized or "svr" in normalized:
            return "kernel"
        return "other"

    def _build_candidate_reasoning(
        self,
        model_name: str,
        class_balance: float,
        task_type: str,
        analysis: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate concise per-candidate rationale."""
        reasons: list[str] = []
        family = self._resolve_model_family(model_name)

        reasons.append("Included as one of the equally weighted candidates for this dataset.")

        if family in {"tree_ensemble", "boosted_trees"}:
            reasons.append("Handles non-linear interactions and mixed feature behavior well.")
        elif family == "linear":
            reasons.append("Provides a regularized, interpretable baseline.")
        elif family == "kernel":
            reasons.append("Can model non-linear boundaries on moderate-size datasets.")

        if task_type == "classification" and class_balance < 0.1:
            reasons.append("Suitable for imbalanced classification with weighted settings.")

        if analysis:
            if analysis.get("has_high_correlations") and family in {"tree_ensemble", "boosted_trees", "linear"}:
                reasons.append("Robust to correlated predictors.")
            if analysis.get("has_outliers") and family in {"tree_ensemble", "boosted_trees"}:
                reasons.append("Less sensitive to outliers than margin-based methods.")

        return " ".join(dict.fromkeys(reasons))

    def _get_search_space_for_model(self, model_name: str) -> dict[str, Any]:
        """Return tuning search space for a model when available."""
        try:
            from core.hyperparameter_optimizer import get_search_space

            return get_search_space(model_name)
        except Exception:
            return {}

    def _sanitize_search_space(self, search_space: dict[str, Any]) -> dict[str, Any]:
        """Keep valid search space entries only."""
        sanitized: dict[str, Any] = {}
        for key, value in search_space.items():
            if not isinstance(value, dict):
                continue
            space_type = str(value.get("type", "")).lower()
            if space_type not in {"int", "float", "categorical"}:
                continue

            if space_type in {"int", "float"}:
                low = value.get("low")
                high = value.get("high")
                if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                    continue
                if low >= high:
                    continue
                sanitized[key] = {
                    "type": space_type,
                    "low": low,
                    "high": high,
                    "log": bool(value.get("log", False)),
                }
                if "step" in value and isinstance(value["step"], (int, float)):
                    sanitized[key]["step"] = value["step"]
            elif space_type == "categorical":
                choices = value.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                sanitized[key] = {
                    "type": "categorical",
                    "choices": choices,
                }

        return sanitized

    def _sanitize_hyperparameters(
        self,
        model_name: str,
        hyperparameters: dict[str, Any],
        task_type: str,
    ) -> dict[str, Any]:
        """Keep only safe, supported hyperparameters for the selected model."""
        allowed_keys: dict[str, set[str]] = {
            "RandomForest": {"random_state", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "class_weight"},
            "GradientBoosting": {"random_state", "n_estimators", "max_depth", "learning_rate", "subsample"},
            "XGBoost": {"random_state", "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "eval_metric", "min_child_weight", "gamma"},
            "LightGBM": {"random_state", "n_estimators", "max_depth", "learning_rate", "num_leaves", "subsample", "colsample_bytree", "min_child_samples", "verbose"},
            "LogisticRegression": {"random_state", "max_iter", "C", "class_weight", "solver", "penalty"},
            "Ridge": {"alpha", "random_state"},
            "SVM": {"C", "kernel", "gamma", "class_weight", "probability"},
            "SVR": {"C", "kernel", "gamma", "epsilon"},
        }

        key = model_name
        if model_name == "SVM" and task_type != "classification":
            key = "SVR"

        filtered = {}
        for param_name, value in hyperparameters.items():
            if param_name not in allowed_keys.get(key, set()):
                continue
            filtered[param_name] = self._coerce_hyperparameter_value(value)
        return filtered

    def _coerce_hyperparameter_value(self, value: Any) -> Any:
        """Convert JSON-like hyperparameter values to Python scalars."""
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, str):
            lowered = value.lower().strip()
            if lowered in {"true", "false"}:
                return lowered == "true"
            try:
                if "." in lowered:
                    return float(lowered)
                return int(lowered)
            except ValueError:
                return value
        if isinstance(value, list):
            return value
        return str(value)
