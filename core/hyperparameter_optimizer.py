"""Hyperparameter Optimization using Optuna for AutoML Pipeline.

This module provides Bayesian optimization for model hyperparameters using Optuna.
"""

from typing import Any, Callable, Optional

import numpy as np
from sklearn.model_selection import cross_val_score

from agents.training_agent import TrainingAgent
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import optuna  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    optuna = None


class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimizer for sklearn models.

    Supports:
    - XGBoost
    - RandomForest
    - GradientBoosting
    - LogisticRegression
    - Ridge
    - LightGBM
    """

    def __init__(
        self,
        n_trials: int = 20,
        timeout: Optional[float] = None,
        cv: int = 5,
        scoring: str = "accuracy",
        random_state: int = 42,
    ) -> None:
        """Initialize the optimizer.

        Args:
            n_trials: Number of optimization trials.
            timeout: Maximum time in seconds for optimization.
            cv: Number of cross-validation folds.
            scoring: Scoring metric for cross-validation.
            random_state: Random seed for reproducibility.
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self._training_agent = TrainingAgent()

    def optimize(
        self,
        model_name: str,
        X_train: Any,
        y_train: Any,
        task_type: str = "classification",
        search_space: Optional[dict[str, Any]] = None,
        base_params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            model_name: Name of the model to optimize.
            X_train: Training features.
            y_train: Training labels.
            task_type: Type of task (classification/regression).

        Returns:
            Dictionary containing:
            - best_params: Optimal hyperparameters
            - best_score: Best cross-validation score
            - study: Optuna study object
        """
        if optuna is None:
            raise RuntimeError("Optuna is not installed. Install optuna or disable HPO.")

        optuna.logging.disable_default_handler()
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)

        if not self._training_agent._is_model_available(model_name):
            raise RuntimeError(f"Model '{model_name}' is not available in this environment.")

        if task_type == "regression":
            self.scoring = "r2"

        # Create sampler with random state for reproducibility
        sampler = optuna.samplers.TPESampler(seed=self.random_state)

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"hpo_{model_name}",
        )

        # Define objective function
        def objective(trial: Any) -> float:
            params = self._suggest_params(trial, model_name, task_type, search_space=search_space)
            if isinstance(base_params, dict):
                params = {**base_params, **params}

            # Create model
            model = self._training_agent._create_model(
                model_name=model_name,
                hyperparameters=params,
                task_type=task_type,
            )

            # Run cross-validation
            try:
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=min(self.cv, min(len(X_train) // 10, 5)),
                    scoring=self.scoring,
                    n_jobs=-1,
                )
                return scores.mean()
            except Exception as e:
                logger.warning(f"Trial failed with params {params}: {e}")
                return float("-inf")

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study,
            "n_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        }

    def _suggest_params(
        self,
        trial: Any,
        model_name: str,
        task_type: str,
        search_space: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Suggest hyperparameters for a trial.

        Args:
            trial: Optuna trial object.
            model_name: Name of the model.
            task_type: Type of task.

        Returns:
            Dictionary of hyperparameters.
        """
        if search_space:
            return self._suggest_from_search_space(trial, search_space)

        model_name_lower = model_name.lower().replace("-", "").replace("_", "")

        params = {"random_state": self.random_state}

        if "randomforest" in model_name_lower:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 20)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)
            params["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

        elif "gradientboosting" in model_name_lower or "gradient" in model_name_lower:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
            params["max_depth"] = trial.suggest_int("max_depth", 2, 10)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)

        elif "xgboost" in model_name_lower:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
            params["max_depth"] = trial.suggest_int("max_depth", 2, 12)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
            params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
            params["gamma"] = trial.suggest_float("gamma", 0, 0.5)
            if task_type == "classification":
                params["eval_metric"] = "logloss"

        elif "lightgbm" in model_name_lower:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
            params["max_depth"] = trial.suggest_int("max_depth", 2, 15)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            params["num_leaves"] = trial.suggest_int("num_leaves", 10, 100)
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
            params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 50)

        elif "logistic" in model_name_lower:
            params["C"] = trial.suggest_float("C", 0.001, 100, log=True)
            params["max_iter"] = trial.suggest_int("max_iter", 500, 2000)
            params["solver"] = trial.suggest_categorical("solver", ["lbfgs", "saga"])
            if params["solver"] == "saga":
                params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
            else:
                params["penalty"] = "l2"

        elif "ridge" in model_name_lower:
            params["alpha"] = trial.suggest_float("alpha", 0.001, 100, log=True)

        return params

    def _suggest_from_search_space(
        self,
        trial: Any,
        search_space: dict[str, Any],
    ) -> dict[str, Any]:
        """Suggest parameters from a serialized search space definition."""
        params: dict[str, Any] = {"random_state": self.random_state}

        for param_name, spec in search_space.items():
            if not isinstance(spec, dict):
                continue

            spec_type = str(spec.get("type", "")).lower()
            if spec_type == "int":
                low = spec.get("low")
                high = spec.get("high")
                if not isinstance(low, int) or not isinstance(high, int):
                    continue
                step = spec.get("step") if isinstance(spec.get("step"), int) else 1
                params[param_name] = trial.suggest_int(param_name, low, high, step=step)

            elif spec_type == "float":
                low = spec.get("low")
                high = spec.get("high")
                if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                    continue
                log = bool(spec.get("log", False))
                if isinstance(spec.get("step"), (int, float)) and not log:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        float(low),
                        float(high),
                        step=float(spec["step"]),
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        float(low),
                        float(high),
                        log=log,
                    )

            elif spec_type == "categorical":
                choices = spec.get("choices")
                if not isinstance(choices, list) or len(choices) == 0:
                    continue
                params[param_name] = trial.suggest_categorical(param_name, choices)

        return params


def get_search_space(model_name: str) -> dict[str, Any]:
    """Get the Optuna search space for a model.

    Args:
        model_name: Name of the model.

    Returns:
        Dictionary describing the search space.
    """
    model_name_lower = model_name.lower().replace("-", "").replace("_", "")

    if "randomforest" in model_name_lower:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 300},
            "max_depth": {"type": "int", "low": 3, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
        }
    elif "gradientboosting" in model_name_lower:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 300},
            "max_depth": {"type": "int", "low": 2, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        }
    elif "xgboost" in model_name_lower:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 300},
            "max_depth": {"type": "int", "low": 2, "high": 12},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        }
    elif "logistic" in model_name_lower:
        return {
            "C": {"type": "float", "low": 0.001, "high": 100, "log": True},
            "max_iter": {"type": "int", "low": 500, "high": 2000},
        }
    elif "ridge" in model_name_lower:
        return {
            "alpha": {"type": "float", "low": 0.001, "high": 100, "log": True},
        }

    return {}
