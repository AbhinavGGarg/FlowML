"""Structured OpenRouter explanations for the evaluation dashboard."""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from config import settings
from utils.openrouter_client import OpenRouterClient

DEFAULT_OPENROUTER_MODEL = "arcee-ai/trinity-large-preview:free"

EVALUATION_INSIGHTS_SYSTEM_PROMPT = """
You are an ML tutor writing short, accurate dashboard explanations for students.
Use ONLY the structured data you are given.
Do not invent metrics, charts, thresholds, or missing values.
If data is missing, say that briefly and conservatively.
Keep the tone beginner-friendly, polished, and technically correct.
Return ONLY valid JSON with this exact shape:
{
  "stage_summary": "string",
  "about_stage_text": "string",
  "performance_story": "string",
  "loss_explanation": "string",
  "generalization_explanation": "string",
  "cross_validation_explanation": "string",
  "baseline_explanation": "string",
  "deployment_reasoning": {
    "recommendation": "deploy | review | do_not_deploy",
    "confidence": "high | medium | low",
    "reason": "string",
    "risk_note": "string",
    "next_step": "string"
  },
  "metric_tooltips": {
    "r2": "string",
    "rmse": "string",
    "mae": "string",
    "accuracy": "string",
    "f1": "string",
    "roc_auc": "string"
  },
  "chart_explanations": {
    "primary_chart": "string",
    "secondary_chart": "string"
  },
  "beginner_notes": ["string", "string", "string"],
  "learning_questions": ["string", "string", "string"]
}
Each string should be concise and dashboard-ready.
`beginner_notes` must be specific to this run.
Make them 3 short items in this order:
1. what happened in this run
2. what looks strongest or most encouraging
3. what looks weakest, risky, or worth improving next
If loss information is available, use it in either `loss_explanation` or the notes.
""".strip()


def build_evaluation_payload(
    training_result: dict[str, Any],
    evaluation_result: dict[str, Any],
    *,
    target_column: Optional[str],
    technical_logs: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build a compact, structured payload for evaluation explanations."""
    task_type = str(evaluation_result.get("task_type") or "classification")
    metrics = {
        "accuracy": _to_float(evaluation_result.get("accuracy")),
        "precision": _to_float(evaluation_result.get("precision")),
        "recall": _to_float(evaluation_result.get("recall")),
        "f1": _to_float(evaluation_result.get("f1")),
        "roc_auc": _to_float(evaluation_result.get("roc_auc")),
        "r2": _to_float(evaluation_result.get("r2")),
        "mae": _to_float(evaluation_result.get("mae")),
        "mse": _to_float(evaluation_result.get("mse")),
        "rmse": _to_float(evaluation_result.get("rmse")),
    }
    train_score = _to_float(training_result.get("train_score"))
    test_score = _to_float(training_result.get("test_score"))
    cv_scores = _to_float_list(training_result.get("cv_scores"))
    cv_mean = _to_float(training_result.get("best_score"))
    cv_std = _to_float(training_result.get("cv_std"))
    train_loss = _to_float_list(training_result.get("train_loss"))
    val_loss = _to_float_list(training_result.get("val_loss"))
    baseline_metrics = evaluation_result.get("baseline_metrics") if isinstance(evaluation_result.get("baseline_metrics"), dict) else None

    return {
        "task_type": task_type,
        "target_column": target_column,
        "model_name": training_result.get("model_name"),
        "metrics": metrics,
        "generalization": {
            "train_score": train_score,
            "test_score": test_score,
            "gap": round(train_score - test_score, 4) if train_score is not None and test_score is not None else None,
        },
        "cross_validation": {
            "scores": cv_scores,
            "mean": cv_mean,
            "std": cv_std,
        },
        "loss_review": {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_epoch": training_result.get("best_epoch"),
            "final_train_loss": train_loss[-1] if train_loss else None,
            "final_val_loss": val_loss[-1] if val_loss else None,
            "loss_gap": (
                round(val_loss[-1] - train_loss[-1], 4)
                if train_loss and val_loss
                else None
            ),
        },
        "baseline_metrics": baseline_metrics,
        "deployment_decision": evaluation_result.get("deployment_decision"),
        "performance_summary": evaluation_result.get("performance_summary"),
        "chart_context": _build_chart_context(task_type, evaluation_result),
        "technical_log_highlights": (technical_logs or [])[-6:],
    }


def generate_evaluation_insights(
    training_result: dict[str, Any],
    evaluation_result: dict[str, Any],
    *,
    target_column: Optional[str],
    technical_logs: Optional[list[str]] = None,
    require_openrouter: bool = False,
) -> dict[str, Any]:
    """Generate structured dashboard copy, optionally requiring OpenRouter."""
    payload = build_evaluation_payload(
        training_result,
        evaluation_result,
        target_column=target_column,
        technical_logs=technical_logs,
    )
    fallback = build_fallback_evaluation_insights(payload)
    client = OpenRouterClient("EvaluationInsights")

    if not client.is_enabled():
        if require_openrouter:
            raise RuntimeError("OpenRouter is not configured for evaluation insights")
        return {
            **fallback,
            "source": "fallback",
            "llm_used": False,
            "model": settings.model_name or DEFAULT_OPENROUTER_MODEL,
        }

    try:
        llm_result = client.generate_json(
            EVALUATION_INSIGHTS_SYSTEM_PROMPT,
            _build_user_prompt(payload, fallback),
            temperature=0.2,
            max_tokens=1400,
        )
        return {
            **(_normalize_openrouter_insights(llm_result) if require_openrouter else _normalize_insights(llm_result, fallback)),
            "source": "openrouter",
            "llm_used": True,
            "model": settings.model_name or DEFAULT_OPENROUTER_MODEL,
        }
    except Exception as exc:
        if require_openrouter:
            raise RuntimeError(f"OpenRouter evaluation insights failed: {exc}") from exc
        response = {
            **fallback,
            "source": "fallback",
            "llm_used": False,
            "model": settings.model_name or DEFAULT_OPENROUTER_MODEL,
            "error": str(exc),
        }
        return response


def build_fallback_evaluation_insights(payload: dict[str, Any]) -> dict[str, Any]:
    """Build conservative deterministic copy when the LLM is unavailable."""
    task_type = str(payload.get("task_type") or "classification")
    metrics = payload.get("metrics", {})
    generalization = payload.get("generalization", {})
    cv = payload.get("cross_validation", {})
    loss_review = payload.get("loss_review", {})
    baseline = payload.get("baseline_metrics") if isinstance(payload.get("baseline_metrics"), dict) else {}
    chart_context = payload.get("chart_context", {})
    deployment_reasoning = _build_fallback_deployment_reasoning(payload)

    if task_type == "regression":
        target_name = str(payload.get("target_column") or "the target")
        stage_summary = (
            f"The model explains about {_as_percent(metrics.get('r2'))} of the variation in {target_name} "
            f"and typically misses by about {_format_number(metrics.get('rmse'))} on unseen data."
        )
        about_stage = (
            "Evaluation checks how close the model's predicted values are to the real values on data it did not train on."
        )
        performance_story = (
            f"An R2 of {_format_decimal(metrics.get('r2'))} means the model captures a meaningful share of the pattern in the target. "
            f"RMSE and MAE show the typical size of the mistakes in the same units as the target."
        )
        baseline_explanation = _build_regression_baseline_text(metrics, baseline)
        chart_explanations = {
            "primary_chart": (
                f"The predicted-vs-actual chart checks whether points stay close to the ideal diagonal line. "
                f"Residual spread is about {_format_number(chart_context.get('residual_std'))}."
            ),
            "secondary_chart": (
                f"The error distribution shows whether most mistakes are small or whether a few large misses dominate. "
                f"The median absolute error is about {_format_number(chart_context.get('median_absolute_error'))}."
            ),
        }
        beginner_notes = [
            f"This run finished with R2 {_format_decimal(metrics.get('r2'))}, RMSE {_format_number(metrics.get('rmse'))}, and MAE {_format_number(metrics.get('mae'))} on unseen data.",
            _build_regression_good_note(metrics, baseline),
            _build_regression_risk_note(generalization, baseline),
        ]
        learning_questions = [
            "Why do we test a regression model on unseen data?",
            "What is the difference between R2 and RMSE?",
            "Why is it useful to compare RMSE against a simple baseline?",
        ]
    else:
        stage_summary = (
            f"The model predicts the correct class about {_as_percent(metrics.get('accuracy'))} of the time, "
            f"with an F1 score of {_as_percent(metrics.get('f1'))} on unseen data."
        )
        about_stage = (
            "Evaluation checks how often the model predicts the right class and whether its mistakes are balanced across the labels."
        )
        performance_story = (
            f"Accuracy shows overall correctness, while F1 balances precision and recall. "
            f"ROC-AUC of {_format_decimal(metrics.get('roc_auc'))} helps show how well the model separates classes when score data is available."
        )
        baseline_explanation = _build_classification_baseline_text(metrics, baseline)
        chart_explanations = {
            "primary_chart": (
                f"The confusion matrix shows which classes the model gets right most often and where it mixes them up. "
                f"The largest off-diagonal count is {_format_number(chart_context.get('largest_error_count'))}."
            ),
            "secondary_chart": (
                f"The class metrics and confidence view show whether performance is balanced across labels. "
                f"Average prediction confidence is about {_as_percent(chart_context.get('average_confidence'))}."
            ),
        }
        beginner_notes = [
            f"This run finished with accuracy {_as_percent(metrics.get('accuracy'))}, F1 {_as_percent(metrics.get('f1'))}, and ROC-AUC {_format_decimal(metrics.get('roc_auc'))} on unseen data.",
            _build_classification_good_note(metrics, baseline),
            _build_classification_risk_note(generalization, chart_context),
        ]
        learning_questions = [
            "Why can accuracy be misleading on an imbalanced dataset?",
            "What does F1 tell you that accuracy alone does not?",
            "Why does beating a majority-class baseline matter?",
        ]

    return {
        "stage_summary": stage_summary,
        "about_stage_text": about_stage,
        "performance_story": performance_story,
        "loss_explanation": _build_loss_text(loss_review),
        "generalization_explanation": _build_generalization_text(generalization),
        "cross_validation_explanation": _build_cv_text(cv),
        "baseline_explanation": baseline_explanation,
        "deployment_reasoning": deployment_reasoning,
        "metric_tooltips": _default_metric_tooltips(),
        "chart_explanations": chart_explanations,
        "beginner_notes": beginner_notes,
        "learning_questions": learning_questions,
    }


def _build_user_prompt(payload: dict[str, Any], fallback: dict[str, Any]) -> str:
    return (
        "Create concise evaluation dashboard copy for this ML run.\n"
        "Use the provided data only.\n"
        "If a field is missing, acknowledge that briefly instead of inventing details.\n"
        "Use this fallback content as a style and safety reference if needed.\n\n"
        f"Structured evaluation payload:\n{json.dumps(payload, indent=2, ensure_ascii=True)}\n\n"
        f"Safe fallback reference:\n{json.dumps(fallback, indent=2, ensure_ascii=True)}"
    )


def _normalize_insights(payload: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    deployment = payload.get("deployment_reasoning") if isinstance(payload.get("deployment_reasoning"), dict) else {}
    fallback_deployment = fallback.get("deployment_reasoning", {})
    metric_tooltips = payload.get("metric_tooltips") if isinstance(payload.get("metric_tooltips"), dict) else {}
    fallback_tooltips = fallback.get("metric_tooltips", {})
    chart_explanations = payload.get("chart_explanations") if isinstance(payload.get("chart_explanations"), dict) else {}
    fallback_chart_explanations = fallback.get("chart_explanations", {})

    return {
        "stage_summary": _read_text(payload, "stage_summary", fallback),
        "about_stage_text": _read_text(payload, "about_stage_text", fallback),
        "performance_story": _read_text(payload, "performance_story", fallback),
        "loss_explanation": _read_text(payload, "loss_explanation", fallback),
        "generalization_explanation": _read_text(payload, "generalization_explanation", fallback),
        "cross_validation_explanation": _read_text(payload, "cross_validation_explanation", fallback),
        "baseline_explanation": _read_text(payload, "baseline_explanation", fallback),
        "deployment_reasoning": {
            "recommendation": _normalize_recommendation(deployment.get("recommendation") or fallback_deployment.get("recommendation")),
            "confidence": _normalize_confidence(deployment.get("confidence") or fallback_deployment.get("confidence")),
            "reason": str(deployment.get("reason") or fallback_deployment.get("reason") or ""),
            "risk_note": str(deployment.get("risk_note") or fallback_deployment.get("risk_note") or ""),
            "next_step": str(deployment.get("next_step") or fallback_deployment.get("next_step") or ""),
        },
        "metric_tooltips": {
            "r2": str(metric_tooltips.get("r2") or fallback_tooltips.get("r2") or ""),
            "rmse": str(metric_tooltips.get("rmse") or fallback_tooltips.get("rmse") or ""),
            "mae": str(metric_tooltips.get("mae") or fallback_tooltips.get("mae") or ""),
            "accuracy": str(metric_tooltips.get("accuracy") or fallback_tooltips.get("accuracy") or ""),
            "f1": str(metric_tooltips.get("f1") or fallback_tooltips.get("f1") or ""),
            "roc_auc": str(metric_tooltips.get("roc_auc") or fallback_tooltips.get("roc_auc") or ""),
        },
        "chart_explanations": {
            "primary_chart": str(chart_explanations.get("primary_chart") or fallback_chart_explanations.get("primary_chart") or ""),
            "secondary_chart": str(chart_explanations.get("secondary_chart") or fallback_chart_explanations.get("secondary_chart") or ""),
        },
        "beginner_notes": _normalize_string_list(payload.get("beginner_notes"), fallback.get("beginner_notes")),
        "learning_questions": _normalize_string_list(payload.get("learning_questions"), fallback.get("learning_questions")),
    }


def _normalize_openrouter_insights(payload: dict[str, Any]) -> dict[str, Any]:
    deployment = payload.get("deployment_reasoning") if isinstance(payload.get("deployment_reasoning"), dict) else {}
    metric_tooltips = payload.get("metric_tooltips") if isinstance(payload.get("metric_tooltips"), dict) else {}
    chart_explanations = payload.get("chart_explanations") if isinstance(payload.get("chart_explanations"), dict) else {}

    return {
        "stage_summary": _read_optional_text(payload, "stage_summary"),
        "about_stage_text": _read_optional_text(payload, "about_stage_text"),
        "performance_story": _read_optional_text(payload, "performance_story"),
        "loss_explanation": _read_optional_text(payload, "loss_explanation"),
        "generalization_explanation": _read_optional_text(payload, "generalization_explanation"),
        "cross_validation_explanation": _read_optional_text(payload, "cross_validation_explanation"),
        "baseline_explanation": _read_optional_text(payload, "baseline_explanation"),
        "deployment_reasoning": {
            "recommendation": _normalize_recommendation(deployment.get("recommendation")),
            "confidence": _normalize_confidence(deployment.get("confidence")),
            "reason": str(deployment.get("reason") or "").strip(),
            "risk_note": str(deployment.get("risk_note") or "").strip(),
            "next_step": str(deployment.get("next_step") or "").strip(),
        },
        "metric_tooltips": {
            "r2": str(metric_tooltips.get("r2") or "").strip(),
            "rmse": str(metric_tooltips.get("rmse") or "").strip(),
            "mae": str(metric_tooltips.get("mae") or "").strip(),
            "accuracy": str(metric_tooltips.get("accuracy") or "").strip(),
            "f1": str(metric_tooltips.get("f1") or "").strip(),
            "roc_auc": str(metric_tooltips.get("roc_auc") or "").strip(),
        },
        "chart_explanations": {
            "primary_chart": str(chart_explanations.get("primary_chart") or "").strip(),
            "secondary_chart": str(chart_explanations.get("secondary_chart") or "").strip(),
        },
        "beginner_notes": _normalize_string_list(payload.get("beginner_notes"), None),
        "learning_questions": _normalize_string_list(payload.get("learning_questions"), None),
    }


def _build_chart_context(task_type: str, evaluation_result: dict[str, Any]) -> dict[str, Any]:
    if task_type == "regression":
        y_true = np.asarray(evaluation_result.get("y_test") or [], dtype=float)
        predictions = np.asarray(evaluation_result.get("predictions") or [], dtype=float)
        if y_true.size == 0 or predictions.size == 0:
            return {}

        residuals = y_true - predictions
        absolute_errors = np.abs(residuals)
        return {
            "point_count": int(y_true.size),
            "target_min": float(np.min(y_true)),
            "target_max": float(np.max(y_true)),
            "prediction_min": float(np.min(predictions)),
            "prediction_max": float(np.max(predictions)),
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "median_absolute_error": float(np.median(absolute_errors)),
            "p90_absolute_error": float(np.percentile(absolute_errors, 90)),
        }

    matrix = evaluation_result.get("confusion_matrix") if isinstance(evaluation_result.get("confusion_matrix"), list) else []
    confidence = _to_float_list(evaluation_result.get("prediction_confidence"))
    largest_error_count = 0.0
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list):
            continue
        for column_index, value in enumerate(row):
            if row_index == column_index:
                continue
            largest_error_count = max(largest_error_count, float(value))

    class_report = evaluation_result.get("classification_report") if isinstance(evaluation_result.get("classification_report"), dict) else {}
    class_summaries = []
    for key, value in class_report.items():
        if not isinstance(value, dict) or key in {"accuracy", "macro avg", "weighted avg"}:
            continue
        class_summaries.append(
            {
                "label": str(key),
                "precision": _to_float(value.get("precision")),
                "recall": _to_float(value.get("recall")),
                "f1": _to_float(value.get("f1-score")),
                "support": _to_float(value.get("support")),
            }
        )

    return {
        "largest_error_count": largest_error_count,
        "average_confidence": float(np.mean(confidence)) if confidence else None,
        "low_confidence_rate": (
            float(np.mean(np.asarray(confidence) < 0.6))
            if confidence
            else None
        ),
        "class_summaries": class_summaries[:6],
    }


def _build_generalization_text(generalization: dict[str, Any]) -> str:
    gap = _to_float(generalization.get("gap"))
    train_score = _format_decimal(generalization.get("train_score"))
    test_score = _format_decimal(generalization.get("test_score"))
    if gap is None:
        return "Train and test scores were not both available, so generalization could not be judged confidently."
    if gap <= 0.03:
        return f"Train score {train_score} and test score {test_score} are very close, which suggests the model generalizes well."
    if gap <= 0.1:
        return f"Train score {train_score} is a bit higher than test score {test_score}, so there may be mild overfitting to watch."
    return f"Train score {train_score} is much higher than test score {test_score}, which suggests the model may be overfitting."


def _build_cv_text(cv: dict[str, Any]) -> str:
    std = _to_float(cv.get("std"))
    mean = _format_decimal(cv.get("mean"))
    if std is None:
        return "Cross-validation details were limited, so stability could not be explained fully."
    if std <= 0.03:
        return f"The fold scores stay tightly grouped around {mean}, which suggests stable performance across validation splits."
    if std <= 0.08:
        return f"The fold scores vary a little around {mean}, so the model looks reasonably stable but not perfectly consistent."
    return f"The fold scores vary quite a bit around {mean}, which means performance may depend strongly on the split."


def _build_loss_text(loss_review: dict[str, Any]) -> str:
    train_loss = _to_float_list(loss_review.get("train_loss"))
    val_loss = _to_float_list(loss_review.get("val_loss"))
    if not train_loss or not val_loss:
        return "Loss-curve details were limited, so training dynamics could not be explained fully."

    final_train = _to_float(loss_review.get("final_train_loss"))
    final_val = _to_float(loss_review.get("final_val_loss"))
    gap = _to_float(loss_review.get("loss_gap"))
    best_epoch = loss_review.get("best_epoch")

    if gap is None:
        return "Both train and validation loss were recorded, but the final loss gap could not be summarized clearly."
    if gap <= 0.03:
        return (
            f"Train loss finished near {_format_number(final_train)} and validation loss near {_format_number(final_val)} by epoch {best_epoch}, "
            "so the learning curves stayed close and training looked stable."
        )
    if gap <= 0.1:
        return (
            f"Train loss ended lower than validation loss by about {_format_number(gap)} at epoch {best_epoch}, "
            "which suggests mild overfitting but not a severe breakdown."
        )
    return (
        f"Train loss finished much lower than validation loss by about {_format_number(gap)} at epoch {best_epoch}, "
        "which is a warning sign that the model may be fitting the training data more tightly than the validation pattern."
    )


def _build_regression_baseline_text(metrics: dict[str, Any], baseline: dict[str, Any]) -> str:
    model_rmse = _to_float(metrics.get("rmse"))
    baseline_rmse = _to_float(baseline.get("rmse"))
    if model_rmse is None or baseline_rmse is None:
        return "A baseline comparison was not available, so this score should be judged with extra caution."
    improvement = ((baseline_rmse - model_rmse) / baseline_rmse) * 100 if baseline_rmse else 0.0
    if improvement > 0:
        return f"The model beats a mean-target baseline by lowering RMSE by about {improvement:.1f}%, which shows it learned more than a simple average."
    return "The model does not clearly beat the mean-target baseline yet, so it may need more feature or model work."


def _build_classification_baseline_text(metrics: dict[str, Any], baseline: dict[str, Any]) -> str:
    model_accuracy = _to_float(metrics.get("accuracy"))
    baseline_accuracy = _to_float(baseline.get("accuracy"))
    if model_accuracy is None or baseline_accuracy is None:
        return "A baseline comparison was not available, so this score should be judged with extra caution."
    improvement = (model_accuracy - baseline_accuracy) * 100
    if improvement > 0:
        return f"The model beats the majority-class baseline by about {improvement:.1f} percentage points, which suggests it learned more than the easiest shortcut."
    return "The model does not clearly beat the majority-class baseline yet, so its practical value is still limited."


def _build_regression_good_note(metrics: dict[str, Any], baseline: dict[str, Any]) -> str:
    r2 = _to_float(metrics.get("r2"))
    rmse = _to_float(metrics.get("rmse"))
    baseline_rmse = _to_float(baseline.get("rmse"))
    if r2 is not None and r2 >= 0.8:
        return f"The strongest sign is that the model explains about {r2 * 100:.1f}% of the target variation, which is a strong regression result."
    if rmse is not None and baseline_rmse is not None and rmse < baseline_rmse:
        improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
        return f"A positive sign is that the model beats the simple baseline by reducing RMSE by about {improvement:.1f}%."
    return "The best sign so far is that the model is learning more than a random guess, but its regression strength is still moderate."


def _build_regression_risk_note(generalization: dict[str, Any], baseline: dict[str, Any]) -> str:
    gap = _to_float(generalization.get("gap"))
    baseline_rmse = _to_float(baseline.get("rmse"))
    if gap is not None and gap > 0.1:
        return "The main concern is the larger train-test gap, which suggests the model may be overfitting."
    if baseline_rmse is not None:
        return "The next thing to watch is whether the error is small enough for the real-world tolerance, not just better than the baseline."
    return "The main caution is that one good score does not guarantee the model will stay reliable on future data."


def _build_classification_good_note(metrics: dict[str, Any], baseline: dict[str, Any]) -> str:
    accuracy = _to_float(metrics.get("accuracy"))
    f1 = _to_float(metrics.get("f1"))
    baseline_accuracy = _to_float(baseline.get("accuracy"))
    if accuracy is not None and f1 is not None and min(accuracy, f1) >= 0.8:
        return "The strongest sign is that both accuracy and F1 are high, so the classifier is not relying on only one flattering score."
    if accuracy is not None and baseline_accuracy is not None and accuracy > baseline_accuracy:
        return f"A good sign is that the model beats the majority-class baseline by about {(accuracy - baseline_accuracy) * 100:.1f} points."
    return "The model shows some class-separation ability, which is better than a naive shortcut, but the gains are still modest."


def _build_classification_risk_note(generalization: dict[str, Any], chart_context: dict[str, Any]) -> str:
    gap = _to_float(generalization.get("gap"))
    low_confidence_rate = _to_float(chart_context.get("low_confidence_rate"))
    if gap is not None and gap > 0.1:
        return "The biggest concern is the train-test gap, which suggests the classifier may be fitting the training data too closely."
    if low_confidence_rate is not None and low_confidence_rate >= 0.35:
        return "A caution sign is that many predictions are not very confident, so the model may be hesitant on harder examples."
    return "The main thing to inspect next is whether mistakes are concentrated in one important class instead of being evenly spread."


def _build_fallback_deployment_reasoning(payload: dict[str, Any]) -> dict[str, str]:
    task_type = str(payload.get("task_type") or "classification")
    raw_decision = str(payload.get("deployment_decision") or "")
    metrics = payload.get("metrics", {})
    generalization = payload.get("generalization", {})
    gap = _to_float(generalization.get("gap"))

    recommendation = "review"
    confidence = "medium"
    reason = "The model shows some useful signal, but it should be checked carefully before deployment."
    risk_note = "Review failure cases and monitor for drift once new data arrives."
    next_step = "Inspect the errors and compare the model against a stronger baseline."

    if task_type == "regression":
        r2 = _to_float(metrics.get("r2")) or 0.0
        if raw_decision == "deploy" or r2 >= 0.8:
            recommendation = "deploy"
            confidence = "high" if gap is not None and gap <= 0.08 else "medium"
            reason = "The regression scores are strong enough to consider deployment, especially if the error size fits the real-world tolerance."
            risk_note = "Check whether the remaining error is acceptable for the target's unit and business impact."
            next_step = "Validate the model on a fresh sample from production-like data."
        elif raw_decision == "reject" or r2 < 0.5:
            recommendation = "do_not_deploy"
            confidence = "high"
            reason = "The model does not explain enough of the target variation yet to be a dependable deployed system."
            risk_note = "Large prediction errors may lead to unreliable decisions."
            next_step = "Improve features, try a better model family, or gather cleaner data."
    else:
        accuracy = _to_float(metrics.get("accuracy")) or 0.0
        f1 = _to_float(metrics.get("f1")) or 0.0
        if raw_decision == "deploy" or (accuracy >= 0.8 and f1 >= 0.8 and (gap is None or gap <= 0.12)):
            recommendation = "deploy"
            confidence = "high" if gap is not None and gap <= 0.08 else "medium"
            reason = "The classification scores are strong enough to consider deployment, with a reasonable balance between correctness and class coverage."
            risk_note = "Check whether any specific class still has weaker recall or precision before using the model in a high-stakes setting."
            next_step = "Validate the model on a fresh holdout sample and monitor confusion hot spots after launch."
        elif raw_decision == "reject" or max(accuracy, f1) < 0.6:
            recommendation = "do_not_deploy"
            confidence = "high"
            reason = "The classification performance is still too weak for dependable deployment."
            risk_note = "Important classes may be misclassified too often."
            next_step = "Improve data balance, feature quality, or model choice before another deployment review."

    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "reason": reason,
        "risk_note": risk_note,
        "next_step": next_step,
    }


def _default_metric_tooltips() -> dict[str, str]:
    return {
        "r2": "R2 shows how much of the target's variation the model explains. Higher is usually better.",
        "rmse": "RMSE is the typical prediction error size, with larger mistakes weighted more strongly.",
        "mae": "MAE is the average absolute prediction error in the same units as the target.",
        "accuracy": "Accuracy is the share of predictions that matched the true class.",
        "f1": "F1 balances precision and recall, so it rewards models that are both correct and complete.",
        "roc_auc": "ROC-AUC measures how well the model separates classes across different decision thresholds.",
    }


def _normalize_string_list(value: Any, fallback: Any) -> list[str]:
    source = value if isinstance(value, list) else fallback
    if not isinstance(source, list):
        return []
    items = [str(item).strip() for item in source if str(item).strip()]
    return items[:4]


def _normalize_recommendation(value: Any) -> str:
    normalized = str(value or "review").strip().lower()
    if normalized in {"deploy", "review", "do_not_deploy"}:
        return normalized
    if normalized in {"reject", "do-not-deploy", "do not deploy"}:
        return "do_not_deploy"
    if normalized in {"iterate", "needs_review"}:
        return "review"
    return "review"


def _normalize_confidence(value: Any) -> str:
    normalized = str(value or "medium").strip().lower()
    if normalized in {"high", "medium", "low"}:
        return normalized
    return "medium"


def _read_text(payload: dict[str, Any], key: str, fallback: dict[str, Any]) -> str:
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    fallback_value = fallback.get(key)
    return str(fallback_value or "").strip()


def _read_optional_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value.strip()
    return ""


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not np.isnan(value):
        return float(value)
    return None


def _to_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    items: list[float] = []
    for item in value:
        parsed = _to_float(item)
        if parsed is not None:
            items.append(parsed)
    return items


def _format_decimal(value: Any) -> str:
    parsed = _to_float(value)
    return f"{parsed:.3f}" if parsed is not None else "not available"


def _format_number(value: Any) -> str:
    parsed = _to_float(value)
    return f"{parsed:.3f}" if parsed is not None else "not available"


def _as_percent(value: Any) -> str:
    parsed = _to_float(value)
    return f"{parsed * 100:.1f}%" if parsed is not None else "not available"
