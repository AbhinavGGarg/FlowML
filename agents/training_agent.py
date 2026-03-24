"""Training Agent for AutoML Pipeline.

This agent handles model training including:
- Cross-validation
- Hyperparameter tuning with Optuna
- Multi-model comparison
- Real training/validation loss tracking
- Ensemble support
"""

import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC, SVR

from agents.base_agent import BaseAgent
from core.exceptions import AgentExecutionError
from utils.logger import get_logger

logger = get_logger(__name__)


class TrainingAgent(BaseAgent):
    """Agent for training machine learning models.

    This agent handles:
    - Model training with cross-validation
    - Hyperparameter optimization with Optuna
    - Multi-model comparison
    - Real training history tracking
    - Ensemble training
    """

    def __init__(self) -> None:
        """Initialize the TrainingAgent."""
        super().__init__("Training")

    async def execute(
        self,
        df: pd.DataFrame,
        model_selection: dict[str, Any],
        pipeline_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Train the selected model.

        Args:
            df: The input DataFrame.
            model_selection: Model selection results from ModelSelectionAgent.
            pipeline_config: Pipeline configuration including test_size, random_state.

        Returns:
            Dictionary containing training results including:
            - model
            - best_score
            - cv_scores
            - train_loss, val_loss
            - best_epoch
            - training_time
            - model_comparisons (if multi-model enabled)
        """
        try:
            test_size = pipeline_config.get("test_size", 0.2)
            random_state = pipeline_config.get("random_state", 42)
            preprocessing_result = pipeline_config.get("preprocessing_result", {})
            training_overrides = pipeline_config.get("training_overrides", {})

            # Get target column from model selection
            target_column = model_selection.get("target_column", df.columns[-1])
            task_type = model_selection.get("task_type", "classification")
            selected_features = model_selection.get("selected_features", [])
            engineered_df = model_selection.get("_engineered_df")

            if (
                not isinstance(engineered_df, pd.DataFrame)
                and isinstance(preprocessing_result, dict)
                and isinstance(preprocessing_result.get("_X_train_transformed"), pd.DataFrame)
                and isinstance(preprocessing_result.get("_X_test_transformed"), pd.DataFrame)
            ):
                X_train = preprocessing_result["_X_train_transformed"].copy()
                X_test = preprocessing_result["_X_test_transformed"].copy()
                y_train = preprocessing_result.get("_y_train")
                y_test = preprocessing_result.get("_y_test")
                if not isinstance(y_train, pd.Series) or not isinstance(y_test, pd.Series):
                    raise ValueError("Preprocessing split artifacts are incomplete for training")
            else:
                feature_source = (
                    engineered_df.copy()
                    if isinstance(engineered_df, pd.DataFrame)
                    else df.drop(columns=[target_column]).copy()
                )
                if selected_features:
                    available_features = [
                        column for column in selected_features
                        if column in feature_source.columns
                    ]
                    X = feature_source[available_features].copy()
                else:
                    X = feature_source.copy()
                y = df[target_column].loc[X.index]

                if X.empty:
                    raise ValueError("No features available for training after feature selection")

                train_indices = preprocessing_result.get("_train_indices", []) if isinstance(preprocessing_result, dict) else []
                test_indices = preprocessing_result.get("_test_indices", []) if isinstance(preprocessing_result, dict) else []
                if isinstance(train_indices, list) and isinstance(test_indices, list) and train_indices and test_indices:
                    available_train = [index for index in train_indices if index in X.index]
                    available_test = [index for index in test_indices if index in X.index]
                    if available_train and available_test:
                        X_train = X.loc[available_train].copy()
                        X_test = X.loc[available_test].copy()
                        y_train = y.loc[available_train].copy()
                        y_test = y.loc[available_test].copy()
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state
                        )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )

                X_train, X_test = self._encode_categorical_train_test(X_train, X_test)

            if X_train.empty:
                raise ValueError("No features available for training after feature selection")

            # Determine if we should do multi-model comparison
            enable_multi_model = pipeline_config.get("enable_multi_model", False)
            top_candidates = self._extract_top_candidates(model_selection)
            top_candidates = self._apply_training_overrides(
                top_candidates=top_candidates,
                task_type=task_type,
                n_samples=len(df),
                training_overrides=training_overrides if isinstance(training_overrides, dict) else {},
            )
            candidate_models = [candidate["model_name"] for candidate in top_candidates]

            if enable_multi_model and candidate_models and len(candidate_models) > 1:
                # Multi-model comparison mode
                return await self._train_with_comparison(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    model_selection=model_selection,
                    top_candidates=top_candidates,
                    task_type=task_type,
                    random_state=random_state,
                    pipeline_config=pipeline_config,
                )
            else:
                # Single model training mode (original behavior with real training history)
                return await self._train_single_model(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    model_selection=model_selection,
                    top_candidates=top_candidates,
                    task_type=task_type,
                    random_state=random_state,
                    pipeline_config=pipeline_config,
                )

        except Exception as e:
            raise AgentExecutionError(
                f"Training failed: {str(e)}",
                agent_name=self.name,
                details={"error": str(e)},
            ) from e

    def _encode_categorical_train_test(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Encode non-numeric columns using training categories only."""
        train_encoded = X_train.copy()
        test_encoded = X_test.copy()

        for column in train_encoded.columns:
            if pd.api.types.is_numeric_dtype(train_encoded[column]):
                continue
            train_values = train_encoded[column].astype(str)
            test_values = test_encoded[column].astype(str)
            categories = sorted(train_values.unique().tolist())
            train_encoded[column] = pd.Categorical(train_values, categories=categories).codes
            test_encoded[column] = pd.Categorical(test_values, categories=categories).codes

        return train_encoded, test_encoded

    async def _train_with_comparison(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_selection: dict[str, Any],
        top_candidates: list[dict[str, Any]],
        task_type: str,
        random_state: int,
        pipeline_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Train multiple models and compare them."""
        from core.model_comparator import ModelComparator

        candidate_models = [candidate["model_name"] for candidate in top_candidates]
        optimize_hpo = pipeline_config.get("optimize_hyperparameters", True)
        scoring = self._resolve_scoring_metric(task_type, pipeline_config)
        try:
            from utils.lightgbm_logger import install_lightgbm_warning_counter, reset_lightgbm_warning_counter

            install_lightgbm_warning_counter()
            reset_lightgbm_warning_counter()
        except Exception:
            pass

        comparator = ModelComparator(
            cv_folds=pipeline_config.get("cv_folds", 5),
            n_trials_hpo=pipeline_config.get("n_trials_hpo", 10) if optimize_hpo else 0,
            random_state=random_state,
        )

        comparison_result = await comparator.compare_models(
            X_train=X_train,
            y_train=y_train,
            candidate_models=candidate_models,
            candidate_specs=top_candidates,
            task_type=task_type,
            optimize_hyperparameters=optimize_hpo,
            scoring=scoring,
        )

        # Get best model and train final model
        best_model_name = comparison_result["best_model"]
        best_params = comparison_result["best_params"]
        selected_candidate = self._find_candidate(top_candidates, best_model_name)
        compared_candidates = [
            self._find_candidate(top_candidates, name) or {"model_name": name}
            for name in candidate_models
        ]

        # Create and train final model
        model = self._create_model(best_model_name, best_params, task_type)

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        # Get training history
        train_loss, val_loss, loss_source = self._get_training_history(
            model, X_train, y_train, X_test, y_test, task_type
        )

        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        tree_metrics = self._build_tree_metrics(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task_type=task_type,
        )

        # Check if ensemble should be built
        enable_ensemble = pipeline_config.get("enable_ensemble", False)
        ensemble_result = None
        if enable_ensemble and len(candidate_models) >= 2:
            from core.ensemble_builder import EnsembleBuilder, should_use_ensemble

            if should_use_ensemble(comparison_result["model_comparisons"]):
                builder = EnsembleBuilder(random_state=random_state)
                ensemble_result = builder.build_ensemble_from_results(
                    model_comparisons=comparison_result["model_comparisons"],
                    task_type=task_type,
                    ensemble_type=pipeline_config.get("ensemble_type", "stacking"),
                    top_k=pipeline_config.get("ensemble_top_k", 3),
                )

                # Train ensemble
                ensemble = ensemble_result["ensemble"]
                ensemble.fit(X_train, y_train)

                # Evaluate ensemble
                ensemble_eval = builder.evaluate_ensemble(
                    ensemble, X_train, y_train, task_type, cv=5
                )
                ensemble_result["cv_score"] = ensemble_eval["cv_mean"]
                ensemble_result["cv_std"] = ensemble_eval["cv_std"]

                # If ensemble is better, use it
                if ensemble_eval["cv_mean"] > comparison_result["best_cv_score"]:
                    model = ensemble
                    test_score = model.score(X_test, y_test)
                    best_model_name = f"Ensemble ({', '.join(ensemble_result['base_models'])})"
                    best_params = {}

        return {
            "model": model,
            "model_name": best_model_name,
            "best_score": comparison_result["best_cv_score"],
            "cv_scores": comparison_result["best_model_result"].get("cv_scores", []),
            "cv_std": comparison_result["best_cv_std"],
            "train_score": float(train_score),
            "test_score": float(test_score),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_epoch": self._find_best_epoch(val_loss) if val_loss else 0,
            "loss_source": loss_source,
            "tree_metrics": tree_metrics,
            "feature_count": X_train.shape[1],
            "training_time": float(training_time),
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "selected_features": list(X_train.columns),
            "model_comparisons": comparison_result["model_comparisons"],
            "best_params": best_params,
            "ensemble_result": ensemble_result,
            "candidate_plan": top_candidates,
            "compared_candidates": compared_candidates,
            "selected_candidate": selected_candidate,
            "training_mode": "multi_model" if len(candidate_models) > 1 else "single",
        }

    async def _train_single_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_selection: dict[str, Any],
        top_candidates: list[dict[str, Any]],
        task_type: str,
        random_state: int,
        pipeline_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Train a single model with optional hyperparameter optimization."""
        selected_candidate = top_candidates[0] if top_candidates else {
            "model_name": "RandomForest",
            "fixed_params": {},
            "search_space": {},
        }
        model_name = selected_candidate.get("model_name", "RandomForest")
        hyperparameters = dict(selected_candidate.get("fixed_params") or {})
        search_space = selected_candidate.get("search_space") if isinstance(selected_candidate.get("search_space"), dict) else None
        try:
            from utils.lightgbm_logger import install_lightgbm_warning_counter, reset_lightgbm_warning_counter

            install_lightgbm_warning_counter()
            reset_lightgbm_warning_counter()
        except Exception:
            pass

        # Optionally run HPO
        optimize_hpo = pipeline_config.get("optimize_hyperparameters", False)
        scoring = self._resolve_scoring_metric(task_type, pipeline_config)
        if optimize_hpo and pipeline_config.get("n_trials_hpo", 0) > 0:
            try:
                from core.hyperparameter_optimizer import HyperparameterOptimizer

                optimizer = HyperparameterOptimizer(
                    n_trials=pipeline_config.get("n_trials_hpo", 10),
                    cv=pipeline_config.get("cv_folds", 5),
                    scoring=scoring,
                    random_state=random_state,
                )
                opt_result = optimizer.optimize(
                    model_name,
                    X_train,
                    y_train,
                    task_type,
                    search_space=search_space,
                    base_params=hyperparameters,
                )
                hyperparameters = {**hyperparameters, **opt_result.get("best_params", {})}
                logger.info(f"Optimized {model_name}: best CV = {opt_result.get('best_score', 0):.4f}")
            except Exception as e:
                logger.warning(f"Hyperparameter optimization skipped for {model_name}: {e}")

        # Create model
        model = self._create_model(model_name, hyperparameters, task_type)

        # Get real training history if available (XGBoost, LightGBM, etc.)
        train_loss, val_loss, loss_source = self._get_training_history(
            model, X_train, y_train, X_test, y_test, task_type
        )

        # Cross-validation
        cv_folds = pipeline_config.get("cv_folds", 5)
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=min(cv_folds, len(X_train) // 10),
            scoring=scoring,
        )

        # Train on full training set and time it
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        # Calculate scores
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        tree_metrics = self._build_tree_metrics(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task_type=task_type,
        )

        # If no real training history was captured, generate from CV fold scores
        if not train_loss or len(train_loss) == 0:
            n_epochs = 8
            train_loss = self._simulate_from_cv(n_epochs, train_score, decreasing=True)
            val_loss = self._simulate_from_cv(n_epochs, cv_scores.mean(), decreasing=False)
            loss_source = "simulated"

        return {
            "model": model,
            "model_name": model_name,
            "best_score": float(cv_scores.mean()),
            "cv_scores": cv_scores.tolist(),
            "cv_std": float(cv_scores.std()),
            "train_score": float(train_score),
            "test_score": float(test_score),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_epoch": self._find_best_epoch(val_loss) if val_loss else 0,
            "loss_source": loss_source,
            "tree_metrics": tree_metrics,
            "feature_count": X_train.shape[1],
            "training_time": float(training_time),
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "selected_features": list(X_train.columns),
            "hyperparameters": hyperparameters,
            "candidate_plan": top_candidates,
            "selected_candidate": {
                **selected_candidate,
                "fixed_params": hyperparameters,
            },
            "training_mode": "single",
        }

    def _extract_top_candidates(self, model_selection: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize model selection output to prioritized candidate specs."""
        raw_candidates = model_selection.get("top_candidates")
        if isinstance(raw_candidates, list) and raw_candidates:
            normalized: list[dict[str, Any]] = []
            for index, item in enumerate(raw_candidates[:3]):
                if not isinstance(item, dict):
                    continue
                model_name = str(item.get("model_name") or "").strip()
                if not model_name:
                    continue
                normalized.append(
                    {
                        "priority": index + 1,
                        "model_name": model_name,
                        "model_family": str(item.get("model_family") or "other"),
                        "reasoning": str(item.get("reasoning") or ""),
                        "fixed_params": item.get("fixed_params", {}) if isinstance(item.get("fixed_params"), dict) else {},
                        "search_space": item.get("search_space", {}) if isinstance(item.get("search_space"), dict) else {},
                    }
                )
            if normalized:
                return normalized

        fallback_model = str(model_selection.get("selected_model") or "RandomForest")
        fallback_params = model_selection.get("hyperparameters", {})
        return [
            {
                "priority": 1,
                "model_name": fallback_model,
                "model_family": "other",
                "reasoning": "Fallback candidate generated from legacy selection output.",
                "fixed_params": fallback_params if isinstance(fallback_params, dict) else {},
                "search_space": {},
            }
        ]

    def _apply_training_overrides(
        self,
        *,
        top_candidates: list[dict[str, Any]],
        task_type: str,
        n_samples: int,
        training_overrides: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Adjust candidate order and params using validated training overrides."""
        candidates = [dict(candidate) for candidate in top_candidates]
        if not candidates:
            return candidates

        force_model_name = str(training_overrides.get("force_model_name") or "").strip()
        preferred_family = str(training_overrides.get("preferred_model_family") or "").strip()

        if force_model_name:
            matching = [
                candidate for candidate in candidates
                if self._normalize_model_name(candidate.get("model_name", "")) == self._normalize_model_name(force_model_name)
            ]
            if matching:
                forced = matching[0]
                candidates = [forced] + [candidate for candidate in candidates if candidate is not forced]
            else:
                candidates.insert(
                    0,
                    {
                        "priority": 1,
                        "model_name": force_model_name,
                        "model_family": self._infer_model_family(force_model_name),
                        "reasoning": "Forced by revision agent.",
                        "fixed_params": self._get_default_hyperparameters(force_model_name, n_samples, task_type),
                        "search_space": {},
                    },
                )
        elif preferred_family:
            ranked = [
                candidate for candidate in candidates
                if str(candidate.get("model_family") or "") == preferred_family
            ]
            remaining = [
                candidate for candidate in candidates
                if str(candidate.get("model_family") or "") != preferred_family
            ]
            if ranked:
                candidates = ranked + remaining
            else:
                fallback_name = self._default_model_for_family(preferred_family, task_type)
                candidates.insert(
                    0,
                    {
                        "priority": 1,
                        "model_name": fallback_name,
                        "model_family": preferred_family,
                        "reasoning": "Added by revision agent to satisfy the requested model family.",
                        "fixed_params": self._get_default_hyperparameters(fallback_name, n_samples, task_type),
                        "search_space": {},
                    },
                )

        for candidate in candidates:
            fixed_params = dict(candidate.get("fixed_params") or {})
            model_name = str(candidate.get("model_name") or "")
            fixed_params = self._apply_parameter_overrides(
                model_name=model_name,
                fixed_params=fixed_params,
                training_overrides=training_overrides,
                task_type=task_type,
            )
            candidate["fixed_params"] = fixed_params
            candidate["model_family"] = str(candidate.get("model_family") or self._infer_model_family(model_name))

        for index, candidate in enumerate(candidates[:3], start=1):
            candidate["priority"] = index
        return candidates[:3]

    def _apply_parameter_overrides(
        self,
        *,
        model_name: str,
        fixed_params: dict[str, Any],
        training_overrides: dict[str, Any],
        task_type: str,
    ) -> dict[str, Any]:
        """Apply safe deterministic parameter changes for revisions."""
        params = dict(fixed_params)
        normalized = self._normalize_model_name(model_name)
        if task_type == "classification" and training_overrides.get("enable_class_weights"):
            if "logistic" in normalized or "randomforest" in normalized or normalized in {"svm", "svc"}:
                params["class_weight"] = "balanced"

        regularization = str(training_overrides.get("regularization_strength") or "normal")
        reduce_complexity = bool(training_overrides.get("reduce_complexity", False))

        if "randomforest" in normalized:
            if reduce_complexity:
                params["max_depth"] = min(int(params.get("max_depth", 10)), 6)
                params["min_samples_split"] = max(int(params.get("min_samples_split", 2)), 8)
                params["n_estimators"] = min(int(params.get("n_estimators", 100)), 80)
        if "gradientboosting" in normalized:
            if reduce_complexity:
                params["max_depth"] = min(int(params.get("max_depth", 5)), 3)
                params["n_estimators"] = min(int(params.get("n_estimators", 100)), 80)
                params["learning_rate"] = min(float(params.get("learning_rate", 0.1)), 0.08)
        if "logisticregression" in normalized:
            if regularization in {"high", "very_high"}:
                params["C"] = 0.5 if regularization == "high" else 0.25
                params.setdefault("solver", "lbfgs")
                params.setdefault("max_iter", 1000)
        if normalized in {"svm", "svc", "svr"} and regularization in {"high", "very_high"}:
            params["C"] = 0.7 if regularization == "high" else 0.4
        if "ridge" in normalized and regularization in {"high", "very_high"}:
            params["alpha"] = 2.0 if regularization == "high" else 4.0

        return params

    def _resolve_scoring_metric(self, task_type: str, pipeline_config: dict[str, Any]) -> str:
        """Map metric priority into a scoring string when possible."""
        if task_type != "classification":
            return "r2"
        metric = str((pipeline_config.get("training_overrides", {}) or {}).get("metric_priority") or "").strip().lower()
        return {
            "recall": "recall_weighted",
            "precision": "precision_weighted",
            "f1": "f1_weighted",
            "accuracy": "accuracy",
        }.get(metric, "accuracy")

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model names for deterministic matching."""
        return str(model_name).lower().replace("-", "").replace("_", "")

    def _infer_model_family(self, model_name: str) -> str:
        """Infer a broad model family from the model name."""
        normalized = self._normalize_model_name(model_name)
        if "randomforest" in normalized:
            return "tree_ensemble"
        if "gradientboosting" in normalized or "xgboost" in normalized or "lightgbm" in normalized:
            return "boosted_trees"
        if "logistic" in normalized or "ridge" in normalized:
            return "linear"
        if "svm" in normalized or "svr" in normalized or "svc" in normalized:
            return "kernel"
        return "other"

    def _default_model_for_family(self, family: str, task_type: str) -> str:
        """Return a safe default model for a requested family."""
        if family == "linear":
            return "LogisticRegression" if task_type == "classification" else "Ridge"
        if family == "tree_ensemble":
            return "RandomForest"
        if family == "boosted_trees":
            return "GradientBoosting"
        if family == "kernel":
            return "SVM" if task_type == "classification" else "SVR"
        return "RandomForest"

    def _find_candidate(
        self,
        candidates: list[dict[str, Any]],
        model_name: str,
    ) -> Optional[dict[str, Any]]:
        """Find candidate spec by model name."""
        normalized = model_name.lower().replace("-", "").replace("_", "")
        for candidate in candidates:
            candidate_name = str(candidate.get("model_name", "")).lower().replace("-", "").replace("_", "")
            if candidate_name == normalized:
                return candidate
        return None

    def _get_training_history(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str,
    ) -> tuple[list[float], list[float], str]:
        """Get training history if available from the model.

        Supports:
        - XGBoost (evals_result)
        - LightGBM (evals_result_)
        - GradientBoosting (staged_predict) as a proxy

        Returns:
            Tuple of (train_loss, val_loss, loss_source).
        """
        model_name = model.__class__.__name__.lower()

        # Try XGBoost
        if hasattr(model, "evals_result"):
            try:
                evals = model.evals_result()
                if evals:
                    train_key = "train"
                    val_key = "validation" if "validation" in evals else list(evals.keys())[-1]

                    if train_key in evals and val_key in evals:
                        # Get the metric name
                        metric_train = list(evals[train_key].keys())[0] if evals[train_key] else None
                        metric_val = list(evals[val_key].keys())[0] if evals[val_key] else None

                        if metric_train and metric_val:
                            train_loss = [float(v) for v in evals[train_key][metric_train]]
                            val_loss = [float(v) for v in evals[val_key][metric_val]]
                            return train_loss, val_loss, "real"
            except Exception:
                pass

        # Try LightGBM
        if hasattr(model, "evals_result_"):
            try:
                evals = model.evals_result_
                if evals:
                    train_key = "training"
                    val_key = "valid_0" if "valid_0" in evals else list(evals.keys())[-1]

                    if train_key in evals and val_key in evals:
                        train_loss = [float(v) for v in evals[train_key]]
                        val_loss = [float(v) for v in evals[val_key]]
                        return train_loss, val_loss, "real"
            except Exception:
                pass

        # Try sklearn GradientBoosting (staged_predict)
        if "gradientboosting" in model_name:
            try:
                train_losses = []
                val_losses = []

                # Get staged predictions for loss curve
                from sklearn.metrics import log_loss, mean_squared_error

                for i, (train_pred, val_pred) in enumerate(
                    zip(
                        model.staged_predict(X_train),
                        model.staged_predict(X_test)
                    )
                ):
                    if i >= 10:  # Limit iterations
                        break
                    if task_type == "classification":
                        train_losses.append(float(log_loss(y_train, train_pred)))
                        val_losses.append(float(log_loss(y_test, val_pred)))
                    else:
                        train_losses.append(float(mean_squared_error(y_train, train_pred)))
                        val_losses.append(float(mean_squared_error(y_test, val_pred)))

                if train_losses and val_losses:
                    # Normalize to 0-1 range for visualization
                    train_losses = self._normalize_losses(train_losses)
                    val_losses = self._normalize_losses(val_losses)
                    return train_losses, val_losses, "proxy"
            except Exception:
                pass

        return [], [], "unavailable"

    def _normalize_losses(self, losses: list[float]) -> list[float]:
        """Normalize loss values to 0-1 range for visualization."""
        if not losses:
            return losses

        min_loss = min(losses)
        max_loss = max(losses)

        if max_loss == min_loss:
            return [0.5] * len(losses)

        return [(loss - min_loss) / (max_loss - min_loss) for loss in losses]

    def _simulate_from_cv(
        self,
        n_epochs: int,
        cv_score: float,
        decreasing: bool = True,
    ) -> list[float]:
        """Generate simulated loss curve from CV score as fallback."""
        if decreasing:
            start = 1.0 - (cv_score * 0.5)
            end = 1.0 - cv_score
        else:
            start = 0.9 - (cv_score * 0.3)
            end = 1.0 - cv_score

        curve = []
        for i in range(n_epochs):
            progress = i / (n_epochs - 1) if n_epochs > 1 else 0
            value = start + (end - start) * progress + np.random.normal(0, 0.02)
            curve.append(max(0.01, min(1.0, value)))

        return curve

    def _create_model(
        self,
        model_name: str,
        hyperparameters: dict[str, Any],
        task_type: str,
    ):
        """Create a model instance based on name and hyperparameters."""
        model_name = model_name.lower().replace("-", "").replace("_", "")

        if "randomforest" in model_name:
            if task_type == "classification":
                return RandomForestClassifier(**hyperparameters)
            else:
                return RandomForestRegressor(**hyperparameters)
        elif "gradientboosting" in model_name or "gradient" in model_name:
            if task_type == "classification":
                return GradientBoostingClassifier(**hyperparameters)
            else:
                return GradientBoostingRegressor(**hyperparameters)
        elif "xgboost" in model_name:
            try:
                import xgboost as xgb
                params = dict(hyperparameters)
                params.setdefault("verbosity", 0)
                if task_type == "classification":
                    return xgb.XGBClassifier(**params)
                else:
                    return xgb.XGBRegressor(**params)
            except (ImportError, ModuleNotFoundError) as e:
                raise RuntimeError(
                    "XGBoost is not available. Install xgboost and the OpenMP runtime (libomp) to use this model."
                ) from e
        elif "lightgbm" in model_name:
            try:
                import lightgbm as lgb
                params = dict(hyperparameters)
                params.setdefault("verbosity", -1)
                params.setdefault("verbose", -1)
                if task_type == "classification":
                    return lgb.LGBMClassifier(**params)
                else:
                    return lgb.LGBMRegressor(**params)
            except (ImportError, ModuleNotFoundError) as e:
                raise RuntimeError(
                    "LightGBM is not available. Install lightgbm to use this model."
                ) from e
        elif "logistic" in model_name:
            return LogisticRegression(**hyperparameters)
        elif "ridge" in model_name:
            return Ridge(**hyperparameters)
        elif "svr" in model_name or "svm" in model_name:
            if task_type == "classification":
                return SVC(**hyperparameters)
            else:
                return SVR(**hyperparameters)
        else:
            # Default to RandomForest
            if task_type == "classification":
                return RandomForestClassifier(**hyperparameters)
            else:
                return RandomForestRegressor(**hyperparameters)

    def _get_default_hyperparameters(
        self,
        model_name: str,
        n_samples: int,
        task_type: str,
    ) -> dict[str, Any]:
        """Get default hyperparameters for a model.

        Delegates to ModelSelectionAgent to keep defaults consistent.
        """
        try:
            from agents.model_selection_agent import ModelSelectionAgent

            selector = ModelSelectionAgent()
            return selector._get_default_hyperparameters(model_name, n_samples, task_type)
        except Exception:
            # Fallback to a minimal safe default
            if "randomforest" in model_name.lower():
                return {"n_estimators": 100, "random_state": 42}
            return {}

    def _is_model_available(self, model_name: str) -> bool:
        """Check whether optional model dependencies are available."""
        name = model_name.lower()
        logger.info(f"Checking model availability: {model_name} (normalized: {name})")
        if "xgboost" in name:
            try:
                import xgboost  # type: ignore  # noqa: F401
                logger.info("XGBoost import successful.")
            except Exception as e:
                logger.error(f"XGBoost import failed: {e}")
                return False
        if "lightgbm" in name:
            try:
                import lightgbm  # type: ignore  # noqa: F401
                logger.info("LightGBM import successful.")
            except Exception as e:
                logger.error(f"LightGBM import failed: {e}")
                return False
        return True

    def _simulate_loss_curve(
        self,
        n_epochs: int,
        final_score: float,
        decreasing: bool = True,
    ) -> list[float]:
        """Simulate a loss curve for visualization (fallback)."""
        if decreasing:
            start = 1.0 - (final_score * 0.5)
            end = 1.0 - final_score
        else:
            start = 0.9 - (final_score * 0.3)
            end = 1.0 - final_score

        curve = []
        for i in range(n_epochs):
            progress = i / (n_epochs - 1)
            value = start + (end - start) * progress + np.random.normal(0, 0.02)
            curve.append(max(0.01, min(1.0, value)))

        return curve

    def _find_best_epoch(self, val_loss: list[float]) -> int:
        """Find the epoch with lowest validation loss."""
        return int(np.argmin(val_loss))

    def _build_tree_metrics(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str,
    ) -> Optional[dict[str, Any]]:
        """Return tree-specific training metrics for the evaluation panel."""
        model_name = model.__class__.__name__.lower()
        is_tree_model = any(
            key in model_name for key in ["randomforest", "gradientboosting", "xgb", "lightgbm"]
        )
        if not is_tree_model:
            return None

        num_trees = None
        if hasattr(model, "n_estimators"):
            try:
                num_trees = int(model.n_estimators)
            except Exception:
                num_trees = None
        if num_trees is None and hasattr(model, "estimators_"):
            try:
                num_trees = len(model.estimators_)
            except Exception:
                num_trees = None

        train_scores: list[float] = []
        val_scores: list[float] = []
        source = "single"

        if "gradientboosting" in model_name and hasattr(model, "staged_predict"):
            try:
                from sklearn.metrics import accuracy_score, r2_score

                source = "staged"
                max_points = 20
                for index, (train_pred, val_pred) in enumerate(
                    zip(model.staged_predict(X_train), model.staged_predict(X_test))
                ):
                    if index >= max_points:
                        break
                    if task_type == "classification":
                        train_scores.append(float(accuracy_score(y_train, train_pred)))
                        val_scores.append(float(accuracy_score(y_test, val_pred)))
                    else:
                        train_scores.append(float(r2_score(y_train, train_pred)))
                        val_scores.append(float(r2_score(y_test, val_pred)))
            except Exception:
                train_scores = []
                val_scores = []

        if not train_scores or not val_scores:
            try:
                train_scores = [float(model.score(X_train, y_train))]
                val_scores = [float(model.score(X_test, y_test))]
            except Exception:
                train_scores = []
                val_scores = []

        score_name = "Accuracy" if task_type == "classification" else "R2"

        return {
            "model_family": model.__class__.__name__,
            "num_trees": num_trees,
            "train_scores": train_scores,
            "val_scores": val_scores,
            "score_name": score_name,
            "source": source,
        }
