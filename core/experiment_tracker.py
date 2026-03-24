"""MLflow Experiment Tracking for AutoML Pipeline.

This module provides experiment tracking using MLflow for:
- Logging parameters, metrics, and artifacts
- Tracking feature engineering steps
- Model comparison results
- Pipeline lineage
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """MLflow-based experiment tracker for AutoML experiments.

    Provides structured logging of:
    - Run metadata
    - Parameters
    - Metrics
    - Artifacts (models, data, plots)
    - Tags for search and filtering
    """

    def __init__(
        self,
        experiment_name: str = "flowml_automl",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ) -> None:
        """Initialize the experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment.
            tracking_uri: MLflow tracking server URI.
            artifact_location: Path to store artifacts.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.artifact_location = artifact_location

        # Configure MLflow
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        # Set experiment
        self._experiment = mlflow.set_experiment(experiment_name)
        self._client = MlflowClient()

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> mlflow.entities.Run:
        """Start a new MLflow run.

        Args:
            run_name: Name for the run.
            tags: Tags for the run.

        Returns:
            MLflow run object.
        """
        run = mlflow.start_run(
            run_name=run_name,
            tags=tags,
            experiment_id=self._experiment.experiment_id,
        )
        return run

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters.

        Args:
            params: Dictionary of parameters to log.
        """
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics.

        Args:
            metrics: Dictionary of metrics to log.
            step: Optional step number.
        """
        flat_metrics = self._flatten_dict(metrics)
        mlflow.log_metrics(flat_metrics, step=step)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log artifacts from a directory.

        Args:
            local_dir: Path to local directory.
            artifact_path: Path within artifacts.
        """
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

    def log_artifact(self, local_file: str, artifact_path: Optional[str] = None) -> None:
        """Log a single artifact file.

        Args:
            local_file: Path to local file.
            artifact_path: Path within artifacts.
        """
        mlflow.log_artifact(local_file, artifact_path=artifact_path)

    def log_model(self, model: Any, artifact_path: str) -> None:
        """Log a model artifact.

        Args:
            model: Model to log (sklearn, xgboost, etc.).
            artifact_path: Path for the model.
        """
        try:
            mlflow.sklearn.log_model(model, artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log model with sklearn: {e}")
            try:
                mlflow.xgboost.log_model(model, artifact_path)
            except Exception as e2:
                logger.warning(f"Failed to log model with xgboost: {e2}")

    def log_feature_engineering(
        self,
        feature_config: dict[str, Any],
        selected_features: list[str],
        dropped_features: list[str],
    ) -> None:
        """Log feature engineering details.

        Args:
            feature_config: Feature engineering configuration.
            selected_features: List of selected features.
            dropped_features: List of dropped features.
        """
        mlflow.log_dict(feature_config, "feature_config.json")
        mlflow.log_text("\n".join(selected_features), "selected_features.txt")
        mlflow.log_text("\n".join(dropped_features), "dropped_features.txt")

    def log_model_comparison(
        self,
        comparison_results: list[dict[str, Any]],
    ) -> None:
        """Log model comparison results.

        Args:
            comparison_results: List of model comparison results.
        """
        # Log comparison as JSON
        mlflow.log_dict(comparison_results, "model_comparison.json")

        # Log best model metrics
        if comparison_results:
            best = max(comparison_results, key=lambda x: x.get("cv_mean", 0))
            mlflow.log_metrics({
                "best_cv_score": best.get("cv_mean", 0),
                "best_cv_std": best.get("cv_std", 0),
            })

    def log_hyperparameter_search(
        self,
        study: Any,
        model_name: str,
    ) -> None:
        """Log hyperparameter optimization results.

        Args:
            study: Optuna study object.
            model_name: Name of the optimized model.
        """
        # Log best parameters
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics({
            f"best_{model_name}_cv_score": study.best_value,
        })

        # Log trial results as JSON
        trials = []
        for trial in study.trials:
            if trial.state.value == "COMPLETE":
                trials.append({
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                })
        mlflow.log_dict(trials, "hpo_trials.json")

    def log_pipeline_result(
        self,
        stage: str,
        result: dict[str, Any],
    ) -> None:
        """Log results from a pipeline stage.

        Args:
            stage: Name of the pipeline stage.
            result: Stage results.
        """
        # Log key metrics
        metrics = {}
        if "best_score" in result:
            metrics[f"{stage}_score"] = result["best_score"]
        if "cv_std" in result:
            metrics[f"{stage}_cv_std"] = result["cv_std"]

        if metrics:
            mlflow.log_metrics(metrics)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run.

        Args:
            key: Tag key.
            value: Tag value.
        """
        mlflow.set_tag(key, value)

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = "_") -> dict:
        """Flatten a nested dictionary.

        Args:
            d: Dictionary to flatten.
            parent_key: Parent key prefix.
            sep: Separator for keys.

        Returns:
            Flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def get_run(self, run_id: str) -> mlflow.entities.Run:
        """Get a run by ID.

        Args:
            run_id: Run ID.

        Returns:
            MLflow run object.
        """
        return self._client.get_run(run_id)

    def search_runs(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100,
    ) -> list[mlflow.entities.Run]:
        """Search for runs.

        Args:
            filter_string: Filter string.
            max_results: Maximum number of results.

        Returns:
            List of runs.
        """
        return mlflow.search_runs(
            experiment_ids=[self._experiment.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
        )


def create_tracker(
    experiment_name: str = "flowml_automl",
    enable: bool = True,
) -> Optional[ExperimentTracker]:
    """Create an experiment tracker if MLflow is available.

    Args:
        experiment_name: Name of the experiment.
        enable: Whether to enable tracking.

    Returns:
        ExperimentTracker or None if MLflow not available.
    """
    if not enable:
        return None

    try:
        return ExperimentTracker(experiment_name=experiment_name)
    except Exception as e:
        logger.warning(f"Failed to create experiment tracker: {e}")
        return None