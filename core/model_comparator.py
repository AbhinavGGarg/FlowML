"""Multi-Model Comparison for AutoML Pipeline.

This module provides parallel training and comparison of multiple candidate models
using cross-validation to select the best model based on both performance and stability.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import get_scorer

from agents.training_agent import TrainingAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelComparator:
    """Compare multiple models using cross-validation.

    Trains candidate models in parallel, compares using CV score and stability,
    and returns the best model with all comparison results.
    """

    def __init__(
        self,
        cv_folds: int = 5,
        n_trials_hpo: int = 15,
        random_state: int = 42,
        parallel_training: bool = True,
    ) -> None:
        """Initialize the comparator.

        Args:
            cv_folds: Number of cross-validation folds.
            n_trials_hpo: Number of Optuna trials for hyperparameter optimization.
            random_state: Random seed for reproducibility.
            parallel_training: Whether to train models in parallel.
        """
        self.cv_folds = cv_folds
        self.n_trials_hpo = n_trials_hpo
        self.random_state = random_state
        self.parallel_training = parallel_training
        self._training_agent = TrainingAgent()

    async def compare_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        candidate_models: list[str],
        candidate_specs: Optional[list[dict[str, Any]]] = None,
        task_type: str = "classification",
        optimize_hyperparameters: bool = True,
        scoring: Optional[str] = None,
    ) -> dict[str, Any]:
        """Compare multiple candidate models using cross-validation.

        Args:
            X_train: Training features.
            y_train: Training labels.
            candidate_models: List of model names to compare.
            task_type: Type of task (classification/regression).
            optimize_hyperparameters: Whether to run hyperparameter optimization.

        Returns:
            Dictionary containing comparison results:
            - model_comparisons: List of model results with scores
            - best_model: Name of the best model
            - best_params: Best hyperparameters for best model
            - best_cv_score: Best cross-validation score
            - best_cv_std: Standard deviation of best model's CV scores
            - training_time: Total comparison time
        """
        start_time = time.perf_counter()

        scoring = scoring or ("accuracy" if task_type == "classification" else "r2")
        cv = min(self.cv_folds, max(3, len(X_train) // 20))

        missing_models = [
            name for name in candidate_models if not self._training_agent._is_model_available(name)
        ]
        if missing_models:
            raise RuntimeError(
                "Required model dependencies missing for: "
                + ", ".join(sorted(set(missing_models)))
            )

        # Prepare data - encode categorical columns
        X_encoded = self._encode_data(X_train)
        candidate_spec_map = self._build_candidate_spec_map(candidate_specs)

        # Train and compare models
        if self.parallel_training and len(candidate_models) > 1:
            results = await self._compare_parallel(
                X_encoded,
                y_train,
                candidate_models,
                candidate_spec_map,
                task_type,
                scoring,
                cv,
                optimize_hyperparameters,
            )
        else:
            results = await self._compare_sequential(
                X_encoded,
                y_train,
                candidate_models,
                candidate_spec_map,
                task_type,
                scoring,
                cv,
                optimize_hyperparameters,
            )

        # Sort by mean score (descending), then by std (ascending) for stability
        results_sorted = sorted(
            results,
            key=lambda x: (x.get("cv_mean", float("-inf")), -x.get("cv_std", float("inf"))),
            reverse=True,
        )

        # Select best model
        best_result = results_sorted[0] if results_sorted else {}

        training_time = time.perf_counter() - start_time

        return {
            "model_comparisons": results_sorted,
            "best_model": best_result.get("model_name", candidate_models[0] if candidate_models else "RandomForest"),
            "best_params": best_result.get("hyperparameters", {}),
            "best_cv_score": best_result.get("cv_mean", 0.0),
            "best_cv_std": best_result.get("cv_std", 0.0),
            "best_model_result": best_result,
            "training_time": float(training_time),
        }

    async def _compare_parallel(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        candidate_models: list[str],
        candidate_spec_map: dict[str, dict[str, Any]],
        task_type: str,
        scoring: str,
        cv: int,
        optimize_hyperparameters: bool,
    ) -> list[dict[str, Any]]:
        """Compare models in parallel using asyncio.gather."""
        tasks = [
            self._train_and_evaluate(
                model_name,
                candidate_spec_map.get(self._normalize_model_name(model_name), {}),
                X_train,
                y_train,
                task_type,
                scoring,
                cv,
                optimize_hyperparameters,
            )
            for model_name in candidate_models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        valid_results = []
        for model_name, result in zip(candidate_models, results):
            if isinstance(result, Exception):
                if "not available" in str(result).lower():
                    raise result
                logger.warning(f"Model {model_name} failed: {result}")
            else:
                valid_results.append(result)

        return valid_results

    async def _compare_sequential(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        candidate_models: list[str],
        candidate_spec_map: dict[str, dict[str, Any]],
        task_type: str,
        scoring: str,
        cv: int,
        optimize_hyperparameters: bool,
    ) -> list[dict[str, Any]]:
        """Compare models sequentially."""
        results = []
        for model_name in candidate_models:
            try:
                result = await self._train_and_evaluate(
                    model_name,
                    candidate_spec_map.get(self._normalize_model_name(model_name), {}),
                    X_train,
                    y_train,
                    task_type,
                    scoring,
                    cv,
                    optimize_hyperparameters,
                )
                results.append(result)
            except Exception as e:
                if "not available" in str(e).lower():
                    raise
                logger.warning(f"Model {model_name} failed: {e}")
        return results

    async def _train_and_evaluate(
        self,
        model_name: str,
        candidate_spec: dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task_type: str,
        scoring: str,
        cv: int,
        optimize_hyperparameters: bool,
    ) -> dict[str, Any]:
        """Train and evaluate a single model."""
        hyperparameters = self._training_agent._get_default_hyperparameters(model_name, len(X_train), task_type)
        fixed_params = candidate_spec.get("fixed_params") if isinstance(candidate_spec.get("fixed_params"), dict) else {}
        hyperparameters = {**hyperparameters, **fixed_params}
        search_space = candidate_spec.get("search_space") if isinstance(candidate_spec.get("search_space"), dict) else None

        # Optionally optimize hyperparameters
        if optimize_hyperparameters and self.n_trials_hpo > 0:
            try:
                from core.hyperparameter_optimizer import HyperparameterOptimizer

                optimizer = HyperparameterOptimizer(
                    n_trials=self.n_trials_hpo,
                    cv=cv,
                    scoring=scoring,
                    random_state=self.random_state,
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
                logger.warning(f"Hyperparameter optimization failed for {model_name}: {e}")

        # Create model
        model = self._training_agent._create_model(model_name, hyperparameters, task_type)

        # Run cross-validation
        cv_start = time.perf_counter()
        try:
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )
        except Exception as e:
            logger.warning(f"CV failed for {model_name}: {e}")
            cv_scores = np.array([0.0])

        cv_time = time.perf_counter() - cv_start

        # Get OOF predictions for diversity analysis
        try:
            oof_preds = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)
        except Exception:
            oof_preds = None

        return {
            "model_name": model_name,
            "hyperparameters": hyperparameters,
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "cv_time": float(cv_time),
            "oof_predictions": oof_preds,
            "model_family": candidate_spec.get("model_family", "other"),
            "reasoning": candidate_spec.get("reasoning", ""),
        }

    def _build_candidate_spec_map(
        self,
        candidate_specs: Optional[list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        """Create normalized lookup map for candidate specifications."""
        if not candidate_specs:
            return {}

        mapping: dict[str, dict[str, Any]] = {}
        for spec in candidate_specs:
            if not isinstance(spec, dict):
                continue
            model_name = str(spec.get("model_name") or "").strip()
            if not model_name:
                continue
            mapping[self._normalize_model_name(model_name)] = spec
        return mapping

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model names for robust lookups."""
        return model_name.lower().replace("-", "").replace("_", "")

    def _encode_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns for model training."""
        X_encoded = X.copy()
        for col in X_encoded.columns:
            if not pd.api.types.is_numeric_dtype(X_encoded[col]):
                values = X_encoded[col].astype(str)
                categories = sorted(values.unique().tolist())
                X_encoded[col] = pd.Categorical(values, categories=categories).codes
        return X_encoded

    def select_best_from_comparison(
        self,
        comparison_results: list[dict[str, Any]],
        selection_criteria: str = "score",
    ) -> dict[str, Any]:
        """Select the best model from comparison results.

        Args:
            comparison_results: List of model comparison results.
            selection_criteria: How to select best - "score", "stable", or "balanced".

        Returns:
            Best model result.
        """
        if not comparison_results:
            raise ValueError("No comparison results provided")

        if selection_criteria == "score":
            # Just pick highest mean score
            return max(comparison_results, key=lambda x: x.get("cv_mean", float("-inf")))

        elif selection_criteria == "stable":
            # Pick lowest variance model among top performers
            top_50 = sorted(comparison_results, key=lambda x: x.get("cv_mean", 0), reverse=True)[: max(1, len(comparison_results) // 2)]
            return min(top_50, key=lambda x: x.get("cv_std", float("inf")))

        elif selection_criteria == "balanced":
            # Combine score and stability
            def balanced_score(result):
                mean = result.get("cv_mean", 0)
                std = result.get("cv_std", 0.1)
                # Higher is better for mean, lower is better for std
                return mean - 0.5 * std  # Penalize variance slightly

            return max(comparison_results, key=balanced_score)

        return comparison_results[0]


def compute_model_diversity(predictions: list[np.ndarray]) -> float:
    """Compute diversity score between model predictions.

    Higher diversity suggests more potential for ensemble improvement.

    Args:
        predictions: List of prediction arrays from different models.

    Returns:
        Diversity score (0-1, higher = more diverse).
    """
    if len(predictions) < 2:
        return 0.0

    n_models = len(predictions)
    n_samples = len(predictions[0])

    # Compute pairwise disagreement
    disagreement_sum = 0.0
    pairs = 0

    for i in range(n_models):
        for j in range(i + 1, n_models):
            if len(predictions[i]) == len(predictions[j]):
                disagree = np.sum(predictions[i] != predictions[j])
                disagreement_sum += disagree / n_samples
                pairs += 1

    if pairs == 0:
        return 0.0

    return disagreement_sum / pairs
