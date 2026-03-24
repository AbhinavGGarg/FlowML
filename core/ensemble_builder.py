"""Model Ensembling for AutoML Pipeline.

This module provides ensemble methods:
- Voting ensemble (average predictions from multiple models)
- Stacking ensemble (meta-learner on base model predictions)
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

from agents.training_agent import TrainingAgent

logger = logging.getLogger(__name__)


class EnsembleBuilder:
    """Build ensemble models from multiple base models.

    Supports:
    - Voting (soft for classification, average for regression)
    - Stacking with meta-learner
    """

    def __init__(self, random_state: int = 42) -> None:
        """Initialize the ensemble builder.

        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self._training_agent = TrainingAgent()

    def create_voting_ensemble(
        self,
        base_models: list[tuple[str, Any]],
        task_type: str = "classification",
        voting: str = "soft" if task_type == "classification" else "average",
    ) -> Any:
        """Create a voting ensemble from base models.

        Args:
            base_models: List of (name, model) tuples.
            task_type: Type of task (classification/regression).
            voting: Type of voting - "soft" (probability) or "hard" (majority).

        Returns:
            VotingClassifier or VotingRegressor.
        """
        if not base_models:
            raise ValueError("No base models provided")

        if task_type == "classification":
            if voting == "soft":
                # Check all models support predict_proba
                estimators = []
                for name, model in base_models:
                    try:
                        # Test if model supports predict_proba
                        _ = getattr(model, "predict_proba", None)
                        estimators.append((name, model))
                    except Exception:
                        logger.warning(f"Model {name} doesn't support soft voting")
                return VotingClassifier(
                    estimators=estimators,
                    voting="soft",
                    n_jobs=-1,
                )
            else:
                return VotingClassifier(
                    estimators=base_models,
                    voting="hard",
                    n_jobs=-1,
                )
        else:
            return VotingRegressor(
                estimators=base_models,
                n_jobs=-1,
            )

    def create_stacking_ensemble(
        self,
        base_models: list[tuple[str, Any]],
        task_type: str = "classification",
        meta_learner: Optional[Any] = None,
        cv: int = 5,
    ) -> Any:
        """Create a stacking ensemble from base models.

        Args:
            base_models: List of (name, model) tuples.
            task_type: Type of task (classification/regression).
            meta_learner: Meta-learner model (default: LogisticRegression/Ridge).
            cv: Number of cross-validation folds for generating out-of-fold predictions.

        Returns:
            StackingClassifier or StackingRegressor.
        """
        if not base_models:
            raise ValueError("No base models provided")

        # Default meta-learner
        if meta_learner is None:
            if task_type == "classification":
                meta_learner = LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state,
                )
            else:
                meta_learner = Ridge(alpha=1.0, random_state=self.random_state)

        if task_type == "classification":
            return StackingClassifier(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=cv,
                stack_method="predict_proba",
                n_jobs=-1,
            )
        else:
            return StackingRegressor(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=cv,
                n_jobs=-1,
            )

    def build_ensemble_from_results(
        self,
        model_comparisons: list[dict[str, Any]],
        task_type: str = "classification",
        ensemble_type: str = "stacking",
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Build an ensemble from model comparison results.

        Args:
            model_comparisons: List of model results from ModelComparator.
            task_type: Type of task (classification/regression).
            ensemble_type: Type of ensemble ("voting" or "stacking").
            top_k: Number of top models to include in ensemble.

        Returns:
            Dictionary containing ensemble model and metadata.
        """
        if not model_comparisons:
            raise ValueError("No model comparison results provided")

        # Select top k models
        sorted_models = sorted(
            model_comparisons,
            key=lambda x: (x.get("cv_mean", 0), -x.get("cv_std", float("inf"))),
            reverse=True,
        )
        top_models = sorted_models[:top_k]

        logger.info(f"Building {ensemble_type} ensemble from top {top_k} models: {[m['model_name'] for m in top_models]}")

        # Create base model tuples
        base_estimators = []
        for model_result in top_models:
            model_name = model_result["model_name"]
            hyperparameters = model_result.get("hyperparameters", {})
            model = self._training_agent._create_model(
                model_name=model_name,
                hyperparameters=hyperparameters,
                task_type=task_type,
            )
            base_estimators.append((model_name, model))

        # Build ensemble
        if ensemble_type == "voting":
            ensemble = self.create_voting_ensemble(base_estimators, task_type)
        else:
            ensemble = self.create_stacking_ensemble(base_estimators, task_type)

        return {
            "ensemble": ensemble,
            "base_models": [m["model_name"] for m in top_models],
            "ensemble_type": ensemble_type,
            "n_base_models": len(top_models),
        }

    def evaluate_ensemble(
        self,
        ensemble: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task_type: str = "classification",
        cv: int = 5,
    ) -> dict[str, Any]:
        """Evaluate ensemble using cross-validation.

        Args:
            ensemble: Ensemble model.
            X_train: Training features.
            y_train: Training labels.
            task_type: Type of task.
            cv: Number of CV folds.

        Returns:
            Dictionary with evaluation results.
        """
        scoring = "accuracy" if task_type == "classification" else "r2"

        try:
            cv_scores = cross_val_score(
                ensemble,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )
            return {
                "cv_mean": float(cv_scores.mean()),
                "cv_std": float(cv_scores.std()),
                "cv_scores": cv_scores.tolist(),
            }
        except Exception as e:
            logger.warning(f"Ensemble evaluation failed: {e}")
            return {
                "cv_mean": 0.0,
                "cv_std": 0.0,
                "cv_scores": [],
                "error": str(e),
            }


def should_use_ensemble(
    model_comparisons: list[dict[str, Any]],
    min_improvement: float = 0.01,
) -> bool:
    """Determine if ensemble would likely improve over single best model.

    Args:
        model_comparisons: List of model comparison results.
        min_improvement: Minimum expected improvement threshold.

    Returns:
        True if ensemble is recommended.
    """
    if len(model_comparisons) < 2:
        return False

    # Sort by score
    sorted_models = sorted(
        model_comparisons,
        key=lambda x: x.get("cv_mean", 0),
        reverse=True,
    )

    if len(sorted_models) < 2:
        return False

    best_score = sorted_models[0].get("cv_mean", 0)
    second_score = sorted_models[1].get("cv_mean", 0)

    # If top models have similar scores and are diverse, ensemble helps
    score_gap = best_score - second_score

    # If gap is small (< min_improvement), ensemble might help
    # Also consider if std is high (unstable model)
    best_std = sorted_models[0].get("cv_std", 0)

    return score_gap < min_improvement or best_std > 0.05