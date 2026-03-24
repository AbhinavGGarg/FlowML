"""Evaluation Agent for AutoML Pipeline.

This agent evaluates trained models for classification and regression.
"""

import logging
from typing import Any, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from agents.base_agent import BaseAgent
from core.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class EvaluationAgent(BaseAgent):
    """Agent for evaluating trained models."""

    def __init__(self) -> None:
        """Initialize the EvaluationAgent."""
        super().__init__("Evaluation")

    async def execute(
        self,
        training_result: dict[str, Any],
        task_type: str = "classification",
        evaluation_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate the trained model.

        Args:
            training_result: Training results from TrainingAgent.
            task_type: Type of task - "classification" or "regression".

        Returns:
            Dictionary containing evaluation results including:
            - classification metrics: accuracy, precision, recall, f1
            - regression metrics: r2, mae, mse, rmse
            - confusion matrix and classification report for classification
            - deployment decision and performance summary
        """
        try:
            logger.info("Evaluating model performance | task_type=%s", task_type)

            model = training_result.get("model")
            X_test = training_result.get("X_test")
            y_test = training_result.get("y_test")

            if model is None or X_test is None or y_test is None:
                raise AgentExecutionError(
                    "Missing training results for evaluation",
                    agent_name=self.name,
                )

            y_pred = model.predict(X_test)
            y_true = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)

            evaluation_config = evaluation_config or {}

            if task_type == "classification":
                result = self._evaluate_classification(model, y_true, y_pred, X_test)
            else:
                result = self._evaluate_regression(y_true, y_pred)

            result["primary_metric"] = (
                str(evaluation_config.get("primary_metric") or "").strip()
                or ("accuracy" if task_type == "classification" else "r2")
            )
            result["deployment_threshold"] = evaluation_config.get("deployment_threshold")

            deployment_decision = self._make_deployment_decision(result, training_result)
            result["deployment_decision"] = deployment_decision
            result["performance_summary"] = self._generate_performance_summary(result)

            if result["task_type"] == "regression":
                logger.info(
                    "Evaluation complete: %s | R2=%.4f | RMSE=%.4f",
                    deployment_decision,
                    result.get("r2", 0.0),
                    result.get("rmse", 0.0),
                )
            else:
                logger.info(
                    "Evaluation complete: %s | accuracy=%.4f | f1=%.4f",
                    deployment_decision,
                    result.get("accuracy", 0.0),
                    result.get("f1", 0.0),
                )
            return result

        except Exception as e:
            logger.exception("Error in evaluation: %s", e)
            raise AgentExecutionError(
                f"Evaluation failed: {str(e)}",
                agent_name=self.name,
                details={"error": str(e)},
            ) from e

    def _evaluate_classification(
        self,
        model: Any,
        y_test: np.ndarray,
        y_pred: Any,
        X_test: Any,
    ) -> dict[str, Any]:
        """Compute classification metrics."""
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        class_labels = [str(label) for label in np.unique(y_test)]
        probabilities, confidences = self._get_classification_confidence(model, X_test)
        roc_auc = self._compute_roc_auc(y_test, probabilities)
        baseline_metrics = self._build_classification_baseline(y_test)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "class_labels": class_labels,
            "predictions": np.asarray(y_pred).tolist(),
            "y_test": y_test.tolist(),
            "prediction_confidence": confidences,
            "baseline_metrics": baseline_metrics,
            "task_type": "classification",
        }

    def _evaluate_regression(self, y_test: np.ndarray, y_pred: Any) -> dict[str, Any]:
        """Compute regression metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        predictions = np.asarray(y_pred, dtype=float)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        residuals = y_test.astype(float) - predictions
        baseline_metrics = self._build_regression_baseline(y_test)

        return {
            "r2": float(r2),
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "predictions": predictions.tolist(),
            "y_test": y_test.astype(float).tolist(),
            "residuals": residuals.tolist(),
            "absolute_errors": np.abs(residuals).tolist(),
            "baseline_metrics": baseline_metrics,
            "task_type": "regression",
        }

    def _get_classification_confidence(self, model: Any, X_test: Any) -> tuple[Optional[np.ndarray], list[float]]:
        """Return class scores and per-row confidence estimates when available."""
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(X_test), dtype=float)
            return probabilities, np.max(probabilities, axis=1).astype(float).tolist()

        if hasattr(model, "decision_function"):
            decision_scores = np.asarray(model.decision_function(X_test), dtype=float)
            if decision_scores.ndim == 1:
                positive = 1.0 / (1.0 + np.exp(-decision_scores))
                probabilities = np.column_stack([1.0 - positive, positive])
            else:
                shifted = decision_scores - decision_scores.max(axis=1, keepdims=True)
                exp_scores = np.exp(shifted)
                totals = np.clip(exp_scores.sum(axis=1, keepdims=True), a_min=1e-9, a_max=None)
                probabilities = exp_scores / totals
            return probabilities, np.max(probabilities, axis=1).astype(float).tolist()

        return None, []

    def _compute_roc_auc(self, y_test: np.ndarray, probabilities: Optional[np.ndarray]) -> Optional[float]:
        """Compute ROC-AUC when probability-like scores are available."""
        if probabilities is None or len(y_test) == 0:
            return None

        try:
            unique_labels = np.unique(y_test)
            if len(unique_labels) <= 1:
                return None
            if len(unique_labels) == 2:
                return float(roc_auc_score(y_test, probabilities[:, -1]))

            encoded = label_binarize(y_test, classes=unique_labels)
            return float(roc_auc_score(encoded, probabilities, multi_class="ovr", average="weighted"))
        except Exception:
            return None

    def _build_classification_baseline(self, y_test: np.ndarray) -> dict[str, Any]:
        """Compare against a majority-class baseline."""
        unique, counts = np.unique(y_test, return_counts=True)
        majority_index = int(np.argmax(counts))
        majority_label = unique[majority_index]
        baseline_pred = np.full(shape=len(y_test), fill_value=majority_label)

        return {
            "strategy": "majority_class",
            "label": str(majority_label),
            "accuracy": float(accuracy_score(y_test, baseline_pred)),
            "f1": float(f1_score(y_test, baseline_pred, average="weighted", zero_division=0)),
            "roc_auc": None,
        }

    def _build_regression_baseline(self, y_test: np.ndarray) -> dict[str, Any]:
        """Compare against a mean-target baseline."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        baseline_value = float(np.mean(y_test))
        baseline_pred = np.full(shape=len(y_test), fill_value=baseline_value, dtype=float)
        baseline_mse = mean_squared_error(y_test, baseline_pred)

        return {
            "strategy": "mean_target",
            "value": baseline_value,
            "r2": float(r2_score(y_test, baseline_pred)),
            "mae": float(mean_absolute_error(y_test, baseline_pred)),
            "mse": float(baseline_mse),
            "rmse": float(np.sqrt(baseline_mse)),
        }

    def _make_deployment_decision(
        self,
        evaluation_result: dict[str, Any],
        training_result: dict[str, Any],
    ) -> str:
        """Make deployment decision based on metrics."""
        task_type = evaluation_result.get("task_type", "classification")

        if task_type == "classification":
            accuracy = evaluation_result.get("accuracy", 0)
            f1 = evaluation_result.get("f1", 0)
            recall = evaluation_result.get("recall", 0)
            train_score = training_result.get("train_score", 0)
            test_score = training_result.get("test_score", 0)
            overfitting_gap = train_score - test_score
            primary_metric = str(evaluation_result.get("primary_metric") or "accuracy")
            threshold = evaluation_result.get("deployment_threshold")

            if threshold is not None:
                metric_value = {
                    "accuracy": accuracy,
                    "f1": f1,
                    "recall": recall,
                }.get(primary_metric, accuracy)
                if metric_value >= float(threshold):
                    return "deploy"
                if metric_value >= float(threshold) * 0.8:
                    return "iterate"
                return "reject"

            if accuracy >= 0.9 and f1 >= 0.9 and overfitting_gap < 0.1:
                return "deploy"
            if accuracy >= 0.7 and f1 >= 0.7:
                if overfitting_gap > 0.15:
                    return "iterate"
                return "deploy"
            if accuracy >= 0.5:
                return "iterate"
            return "reject"

        r2 = evaluation_result.get("r2", 0)
        if r2 >= 0.8:
            return "deploy"
        if r2 >= 0.5:
            return "iterate"
        return "reject"

    def _generate_performance_summary(self, evaluation_result: dict[str, Any]) -> str:
        """Generate human-readable performance summary."""
        task_type = evaluation_result.get("task_type", "classification")

        if task_type == "classification":
            accuracy = evaluation_result.get("accuracy", 0)
            f1 = evaluation_result.get("f1", 0)
            primary_metric = evaluation_result.get("primary_metric", "accuracy")
            decision = evaluation_result.get("deployment_decision", "unknown")
            return (
                f"Model achieved {accuracy:.1%} accuracy and {f1:.1%} F1 score. "
                f"Primary metric: {primary_metric}. Deployment decision: {decision}."
            )

        r2 = evaluation_result.get("r2", 0)
        rmse = evaluation_result.get("rmse", 0)
        decision = evaluation_result.get("deployment_decision", "unknown")
        return f"Model achieved R2 = {r2:.3f} and RMSE = {rmse:.3f}. Deployment decision: {decision}."
