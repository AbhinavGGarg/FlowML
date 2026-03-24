"""FastAPI application for the AutoML Pipeline.

This module provides REST API endpoints for the AutoML pipeline,
allowing the React frontend to upload datasets, configure the pipeline,
and retrieve results.
"""

import json
import logging
import os
import shutil
import uuid
import warnings
from pathlib import Path
from typing import Any, Optional, Union

# Disable joblib memory mapping to prevent resource tracker warnings
os.environ['JOBLIB_MMAP_MODE'] = ''

# Suppress joblib resource tracker warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked .* objects", category=UserWarning)
warnings.filterwarnings("ignore", message="resource_tracker: .*FileNotFoundError", category=UserWarning)

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from agents.chatbot_orchestrator import ChatbotOrchestrator
from config import settings
from core.pipeline_state import PipelineState
from core.revision_history import RevisionHistoryManager
from utils.logger import get_logger
from utils.openrouter_client import OpenRouterClient
from utils.evaluation_insights import generate_evaluation_insights

logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("uvicorn.access").propagate = False

# Ensure outputs directory exists
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Ensure uploads directory exists
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="FlowML AutoML API")
logger = get_logger("api.main")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


pipeline_state = PipelineState()
chat_client = OpenRouterClient("ChatAssistant", model_name=settings.chat_model_name)
revision_history = RevisionHistoryManager()
chatbot_orchestrator = ChatbotOrchestrator()


@app.on_event("shutdown")
async def shutdown_openrouter_client() -> None:
    """Close the shared OpenRouter HTTP client on app shutdown."""
    OpenRouterClient.close_shared_client()


# Request/Response models
class TargetColumnRequest(BaseModel):
    target_column: str


class PipelineConfig(BaseModel):
    task_type: str = "classification"  # or "regression"
    test_size: float = 0.2
    random_state: int = 42


class StageResultResponse(BaseModel):
    stage_id: str
    status: str
    result: Optional[dict[str, Any]] = None


class DatasetSummaryResponse(BaseModel):
    filename: str
    rows: int
    columns: int
    column_names: list[str]
    column_types: dict[str, str]
    missing_values: dict[str, float]
    numeric_summary: Optional[dict[str, Any]] = None


class DatasetPreviewResponse(BaseModel):
    rows: list[dict[str, Any]]
    columns: list[str]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatSelectionContext(BaseModel):
    text: str
    source_label: Optional[str] = None
    surrounding_text: Optional[str] = None


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    selection_context: Optional[ChatSelectionContext] = None
    mode: str = "suggest"


class ChatResponse(BaseModel):
    answer: str
    llm_used: bool = False
    response_mode: str = "llm"
    revision: Optional[dict[str, Any]] = None
class DeploymentReasoningResponse(BaseModel):
    recommendation: str
    confidence: str
    reason: str
    risk_note: str
    next_step: str


class EvaluationInsightsResponse(BaseModel):
    stage_summary: str
    about_stage_text: str
    performance_story: str
    loss_explanation: str
    generalization_explanation: str
    cross_validation_explanation: str
    baseline_explanation: str
    deployment_reasoning: DeploymentReasoningResponse
    metric_tooltips: dict[str, str]
    chart_explanations: dict[str, str]
    beginner_notes: list[str]
    learning_questions: list[str]
    source: str
    llm_used: bool
    model: str
    error: Optional[str] = None


class ChatStatusResponse(BaseModel):
    status: str
    detail: Optional[str] = None
    last_checked_at: Optional[str] = None


CHAT_CONTEXT_OMIT_KEYS = {
    "X_train",
    "X_test",
    "y_train",
    "y_test",
    "model",
    "oof_predictions",
    "deployment_code",
    "package_path",
    "report_path",
    "model_path",
    "metadata_path",
    "correlations",
}


def _truncate_text(value: str, max_chars: int = 500) -> str:
    """Trim oversized strings so chat prompts stay within a reasonable size."""
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}…"


def _compact_chat_value(value: Any, depth: int = 0) -> Any:
    """Recursively compact chat context payloads to avoid oversized prompts."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_chat_text(value, 900 if depth == 0 else 320)
    if isinstance(value, dict):
        max_items = 18 if depth == 0 else 8
        compacted: dict[str, Any] = {}
        kept = 0
        total = 0
        for key, item in value.items():
            key_text = str(key)
            if key_text in CHAT_CONTEXT_OMIT_KEYS or key_text.startswith("_"):
                continue
            total += 1
            if kept >= max_items:
                continue
            compacted[key_text] = _compact_chat_value(item, depth + 1)
            kept += 1
        if total > kept:
            compacted["_truncated"] = f"{total - kept} additional field(s) omitted"
        return compacted
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        max_items = 10 if depth == 0 else 6
        compacted = [_compact_chat_value(item, depth + 1) for item in items[:max_items]]
        if len(items) > max_items:
            compacted.append(f"... {len(items) - max_items} more item(s)")
        return compacted
    return _truncate_chat_text(str(value), 320)


def _truncate_chat_text(value: str, max_chars: int = 500) -> str:
    """ASCII-only truncation helper used by chat context compaction."""
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


chat_runtime_status: dict[str, Optional[str]] = {
    "status": "responsive" if chat_client.is_enabled() else "unavailable",
    "detail": None if chat_client.is_enabled() else "Chat model not configured.",
    "last_checked_at": None,
}


def _set_chat_runtime_status(status: str, detail: Optional[str] = None) -> None:
    """Persist the latest observed chat-model health."""
    chat_runtime_status["status"] = status
    chat_runtime_status["detail"] = detail
    chat_runtime_status["last_checked_at"] = datetime.now(timezone.utc).isoformat()


# Helper functions
def add_log(stage: str, message: str):
    """Add a log message to a stage."""
    if stage in pipeline_state.stage_logs:
        pipeline_state.stage_logs[stage].append(message)


def add_agent_summary_logs(stage: str, result: Optional[dict[str, Any]]):
    """Add LLM or fallback agent summaries to the stage log stream."""
    # if not result:
    #     return
    #
    # summary = result.get("_agent_summary")
    # if not isinstance(summary, dict):
    #     return
    #
    # agent_name = str(summary.get("agent", stage))
    # step_summary = str(summary.get("step_summary", "")).strip()
    # overall_summary = str(summary.get("overall_summary", "")).strip()
    # why = str(summary.get("why", "")).strip()
    # decisions = summary.get("decisions_made", [])
    # llm_used = bool(summary.get("llm_used", False))
    #
    # source_tag = "LLM" if llm_used else "Fallback"
    # if step_summary:
    #     add_log(stage, f"{agent_name} [{source_tag}] summary: {step_summary}")
    # if isinstance(decisions, list):
    #     for decision in decisions[:3]:
    #         decision_text = str(decision).strip()
    #         if decision_text:
    #             add_log(stage, f"{agent_name} decision: {decision_text}")
    # if why:
    #     add_log(stage, f"{agent_name} why: {why}")
    # if overall_summary:
    #     add_log(stage, f"{agent_name} overall: {overall_summary}")
    return


def summarize_dataset(df: Optional[pd.DataFrame]) -> str:
    """Create a compact dataset summary for console logging."""
    if df is None:
        return "dataset=None"

    preview_columns = ", ".join(str(column) for column in df.columns[:6])
    if len(df.columns) > 6:
        preview_columns += ", ..."
    return f"rows={len(df)}, cols={len(df.columns)}, columns=[{preview_columns}]"


def make_json_safe(value: Any) -> Any:
    """Convert common Python, NumPy, and pandas objects to JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return {
            "rows": len(value),
            "columns": list(value.columns),
        }
    return str(value)


def summarize_stage_result(stage: str, result: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return a frontend-safe stage result summary."""
    if result is None:
        return None

    stage_keys: dict[str, list[str]] = {
        "analysis": [
            "row_count",
            "column_count",
            "feature_count",
            "numeric_columns",
            "categorical_columns",
            "missing_values",
            "high_missing_columns",
            "correlations",
            "high_correlation_pairs",
            "outliers",
            "class_distribution",
            "target_column",
            "recommendations",
            "analysis_summary",
            "risk_level",
            "data_quality",
            "quality_flags",
            "llm_used",
        ],
        "training": [
            "model_name",
            "best_score",
            "cv_scores",
            "cv_std",
            "train_score",
            "test_score",
            "train_loss",
            "val_loss",
            "best_epoch",
            "feature_count",
            "training_time",
            "model_comparisons",
            "training_mode",
            "ensemble_result",
        ],
        "model_selection": [
            "top_candidates",
            "selection_reasoning",
            "llm_returned",
            "llm_summary",
            "analysis_signals",
            "class_balance",
            "n_samples",
            "n_features",
            "task_type",
        ],
        "results": [
            "model_path",
            "metadata_path",
            "pipeline_id",
            "deployment_success",
            "deployment_code",
            "package_path",
            "package_ready",
            "report_path",
            "report_ready",
        ],
    }

    selected = {
        key: value for key, value in result.items()
        if not str(key).startswith("_")
    }
    if stage in stage_keys:
        selected = {key: selected.get(key) for key in stage_keys[stage]}

    return make_json_safe(selected)


def format_evaluation_log(result: dict[str, Any]) -> str:
    """Format an evaluation-stage summary for logs."""
    if result.get("task_type") == "regression":
        return f"Evaluation complete: R2 = {result.get('r2', 0):.4f}, RMSE = {result.get('rmse', 0):.4f}"
    return f"Evaluation complete: Accuracy = {result.get('accuracy', 0):.4f}, F1 = {result.get('f1', 0):.4f}"


def build_chat_context() -> dict[str, Any]:
    """Assemble a grounded chat context from pipeline state."""
    df = pipeline_state.dataset
    dataset_summary: dict[str, Any] = {
        "filename": pipeline_state.dataset_filename,
        "rows": int(len(df)) if df is not None else None,
        "columns": int(len(df.columns)) if df is not None else None,
        "column_names": [str(column) for column in df.columns[:25]] if df is not None else [],
        "target_column": pipeline_state.target_column,
        "preview_rows": [],
        "columns_info": [],
        "numeric_columns": [],
        "categorical_columns": [],
    }

    if df is not None:
        preview_df = df.head(3).fillna("")
        dataset_summary["preview_rows"] = preview_df.to_dict(orient="records")
        for column in df.columns[:20]:
            series = df[column]
            is_numeric = bool(pd.api.types.is_numeric_dtype(series))
            column_info = {
                "name": str(column),
                "dtype": str(series.dtype),
                "is_numeric": is_numeric,
                "missing_pct": float(series.isnull().mean()),
                "unique_values": int(series.nunique(dropna=True)),
            }
            dataset_summary["columns_info"].append(column_info)
            if is_numeric:
                dataset_summary["numeric_columns"].append(str(column))
            else:
                dataset_summary["categorical_columns"].append(str(column))

    stage_results = {
        stage: _compact_chat_value(summarize_stage_result(stage, result))
        for stage, result in pipeline_state.stage_results.items()
        if stage in pipeline_state.stage_statuses
    }
    recent_logs = {
        stage: [_truncate_chat_text(message, 220) for message in logs[-3:]]
        for stage, logs in pipeline_state.stage_logs.items()
        if logs
    }
    completed_stages = [
        stage for stage, status in pipeline_state.stage_statuses.items()
        if status == "completed"
    ]
    revision_history_summary = [
        {
            "run_id": record.run_id,
            "parent_run_id": record.parent_run_id,
            "revision_reason": record.revision_reason,
            "changed_stages": record.changed_stages,
            "metrics": record.metrics,
            "created_at": record.created_at,
        }
        for record in pipeline_state.revision_history[-3:]
    ]
    evaluation = pipeline_state.stage_results.get("evaluation", {}) or {}
    training = pipeline_state.stage_results.get("training", {}) or {}
    explanation = pipeline_state.stage_results.get("explanation", {}) or {}

    return {
        "dataset": dataset_summary,
        "stage_statuses": dict(pipeline_state.stage_statuses),
        "completed_stages": completed_stages,
        "stage_results": stage_results,
        "recent_logs": recent_logs,
        "current_run": {
            "run_id": pipeline_state.current_run_id,
            "target_column": pipeline_state.target_column,
            "task_type": pipeline_state.task_type,
            "revision_reason": pipeline_state.revision_history[-1].revision_reason if pipeline_state.revision_history else None,
            "changed_stages": pipeline_state.revision_history[-1].changed_stages if pipeline_state.revision_history else [],
            "metrics": _compact_chat_value(pipeline_state.metrics_summary()),
        },
        "explanation": _compact_chat_value({
            "summary": explanation.get("summary"),
            "pipeline_summary": explanation.get("pipeline_summary"),
            "explanations": explanation.get("explanations"),
        }),
        "pending_revision_plan": make_json_safe(pipeline_state.pending_revision_plan),
        "recent_revisions": make_json_safe(revision_history_summary),
        "metrics": {
            "task_type": evaluation.get("task_type"),
            "accuracy": evaluation.get("accuracy"),
            "precision": evaluation.get("precision"),
            "recall": evaluation.get("recall"),
            "f1": evaluation.get("f1"),
            "r2": evaluation.get("r2"),
            "mae": evaluation.get("mae"),
            "mse": evaluation.get("mse"),
            "rmse": evaluation.get("rmse"),
            "deployment_decision": evaluation.get("deployment_decision"),
            "performance_summary": evaluation.get("performance_summary"),
            "model_name": training.get("model_name"),
            "best_score": training.get("best_score"),
            "test_score": training.get("test_score"),
        },
    }


def _build_chat_prompt(
    *,
    question: str,
    history: list[dict[str, str]],
    context: dict[str, Any],
    selection_context: Optional[dict[str, Any]] = None,
    extra_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a single grounded prompt payload for the chat assistant."""
    history_lines = []
    for message in history[-8:]:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "").strip()
        if content:
            history_lines.append(f"{role}: {content}")

    prompt_context = make_json_safe(context)
    if extra_context:
        prompt_context["request_context"] = make_json_safe(extra_context)

    return {
        "question": question,
        "conversation_history": history_lines,
        "selection_context": make_json_safe(selection_context),
        "pipeline_context": prompt_context,
    }


def generate_chat_answer(
    question: str,
    history: list[dict[str, str]],
    selection_context: Optional[dict[str, Any]] = None,
    extra_context: Optional[dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> tuple[str, bool, str]:
    """Answer a chat question using the LLM, or report that it is unavailable."""
    context = build_chat_context()

    if not chat_client.is_enabled():
        return (
            "The chat model isn’t available right now, so I can’t answer in assistant mode. "
            "Check the OpenRouter configuration and try again.",
            False,
            "unavailable",
        )

    prompt = _build_chat_prompt(
        question=question,
        history=history,
        context=context,
        selection_context=selection_context,
        extra_context=extra_context,
    )
    last_error: Optional[Exception] = None

    try:
        answer = chat_client.generate_text(
            system_prompt=(
                "You are the FlowML learning assistant. "
                "Answer ONLY using the structured pipeline context you are given. "
                "Do not invent dataset columns, models, transformations, or metrics. "
                "Use whatever pipeline evidence is currently available, including completed stages, metrics, recent logs, revision state, and any request_context metadata. "
                "If the answer is not available in the context, say that clearly and mention what is available. "
                "When the user asks why, give the actual reason supported by the context instead of a canned phrase. "
                "Be helpful, specific to the current dataset and run, beginner-friendly, and conversational. "
                "When the user asks about a pipeline stage like preprocessing, explain both the general concept and what happened in this run if that information exists. "
                "If request_context describes a suggested, applied, compared, or undone revision, explain what happened, why it matters, and the relevant pipeline evidence. "
                "If selection_context is provided, use it silently as context. Do NOT say things like 'you highlighted' or restate the source area unless the user asks. "
                "Answer the latest question directly. Do not repeat prior setup unless needed. "
                "If the user asks about one feature, focus on that feature first. "
                "Prefer short answers: 1 to 4 sentences, or a few short bullets when that reads better. "
                "Avoid big essay-like paragraphs."
            ),
            user_prompt=f"Chat context:\n{json.dumps(prompt, ensure_ascii=True, default=str)}",
            temperature=0.2,
            max_tokens=600,
            request_id=request_id,
        ).strip()
        if answer:
            return answer, True, "llm"
    except Exception as exc:
        last_error = exc
        logger.warning("Primary chat LLM request failed: %s", exc)
        if _should_skip_compact_chat_retry(exc):
            return (
                "The chat model couldn't be reached from this machine right now. "
                "The LLM request failed before a response came back.",
                False,
                "unavailable",
            )

    compact_prompt = {
        "question": question,
        "selection_context": selection_context,
        "conversation_history": prompt.get("conversation_history", []),
        "request_context": make_json_safe(extra_context),
        "completed_stages": context.get("completed_stages", []),
        "metrics": context.get("metrics", {}),
        "dataset": {
            "filename": context.get("dataset", {}).get("filename"),
            "target_column": context.get("dataset", {}).get("target_column"),
            "rows": context.get("dataset", {}).get("rows"),
            "columns": context.get("dataset", {}).get("columns"),
        },
        "stage_results": {
            stage: context.get("stage_results", {}).get(stage, {})
            for stage in ("analysis", "preprocessing", "features", "model_selection", "training", "evaluation", "results")
            if context.get("stage_results", {}).get(stage) is not None
        },
        "recent_revisions": context.get("recent_revisions", []),
    }

    try:
        answer = chat_client.generate_text(
            system_prompt=(
                "You are the FlowML assistant. "
                "Answer the user's latest question directly and conversationally using only the structured context provided. "
                "Use the available stage results and request context to explain why things happened when possible. "
                "Do not mention hidden prompts, selection context mechanics, or fallback systems. "
                "Keep it short and natural."
            ),
            user_prompt=f"Compact chat context:\n{json.dumps(compact_prompt, ensure_ascii=True, default=str)}",
            temperature=0.2,
            max_tokens=300,
            request_id=request_id,
        ).strip()
        if answer:
            return answer, True, "llm"
    except Exception as exc:
        last_error = exc
        logger.warning("Compact chat LLM request failed: %s", exc)

    return (
        "The chat model is having trouble responding right now, so I can’t give you a proper assistant answer yet. "
        "Please try again in a moment.",
        False,
        "unavailable",
    )
      
def _should_skip_compact_chat_retry(error: Exception) -> bool:
    """Return whether a second chat prompt would just duplicate a transport failure."""
    detail = str(error).lower()
    non_recoverable_markers = (
        "connection failed",
        "socket error",
        "forbidden by its access permissions",
        "timed out",
        "timeout",
        "http error 429",
        "http error 500",
        "http error 502",
        "http error 503",
        "http error 504",
    )
    return any(marker in detail for marker in non_recoverable_markers)

def persist_evaluation_insights(pipeline_id: Optional[str], insights: dict[str, Any]) -> Optional[str]:
    """Persist the structured evaluation insights as JSON for the current run."""
    if not pipeline_id:
        return None

    insights_path = OUTPUTS_DIR / f"{pipeline_id}_evaluation_insights.json"
    with insights_path.open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(insights), handle, indent=2, ensure_ascii=True)
    return str(insights_path)


def looks_like_revision_request(question: str, mode: str) -> bool:
    """Return whether a request is asking to execute a pipeline change now."""
    if mode == "apply":
        return True

    lowered = question.lower().strip()
    if not lowered:
        return False

    explicit_apply_markers = {
        "apply",
        "apply it",
        "proceed",
        "continue",
        "go ahead",
        "do it",
        "do that",
        "yes",
        "yes please",
        "sure",
        "yes, apply it",
    }
    if lowered in explicit_apply_markers:
        return True

    explanatory_prefixes = (
        "why ",
        "how ",
        "what ",
        "which ",
        "can you explain",
        "could you explain",
        "explain why",
        "tell me why",
    )
    if lowered.startswith(explanatory_prefixes):
        return False

    direct_change_markers = (
        "run without",
        "train without",
        "retrain without",
        "remove ",
        "drop ",
        "exclude ",
        "add ",
        "include ",
        "switch to",
        "use ",
        "undo ",
        "revert ",
        "go back",
        "make the model",
        "change the",
    )
    preference_markers = (
        "i want to",
        "i would like to",
        "please",
        "let's",
        "lets",
    )

    if any(marker in lowered for marker in direct_change_markers):
        return True
    if any(marker in lowered for marker in preference_markers) and any(
        token in lowered for token in ("without ", "remove ", "drop ", "exclude ", "add ", "include ", "switch ")
    ):
        return True

    return False


def maybe_record_revision(reason: str, changed_stages: list[str]) -> None:
    """Persist a revision snapshot when the pipeline has enough state."""
    if pipeline_state.dataset is None or not pipeline_state.target_column:
        return
    if not pipeline_state.stage_results.get("evaluation"):
        return
    if not pipeline_state.revision_history:
        revision_history.commit_run(
            pipeline_state,
            revision_reason=reason,
            changed_stages=changed_stages,
            changed_configs={},
        )

def cleanup_upload() -> bool:
    """Delete the uploaded dataset file from disk for privacy."""
    path = pipeline_state.dataset_path
    if not path:
        return False

    file_path = Path(path)
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info("Upload cleanup: deleted %s", file_path)
        else:
            logger.info("Upload cleanup: file already missing %s", file_path)
    except Exception as exc:  # pragma: no cover - best-effort cleanup
        logger.warning("Upload cleanup failed for %s: %s", file_path, exc)
        return False
    finally:
        pipeline_state.dataset_path = None
    return True


def start_new_pipeline_run() -> str:
    """Generate and assign a fresh pipeline ID for the current run."""
    pipeline_state.pipeline_id = str(uuid.uuid4())
    logger.info("New pipeline run id=%s", pipeline_state.pipeline_id)
    return pipeline_state.pipeline_id


async def run_pipeline_stage(stage: str, config: PipelineConfig):
    """Run a single pipeline stage and update state."""
    pipeline_state.stage_statuses[stage] = "running"
    add_log(stage, f"Starting {stage} stage...")
    logger.info(
        "Starting stage=%s | file=%s | target=%s | task_type=%s | %s",
        stage,
        pipeline_state.dataset_filename or "unknown",
        pipeline_state.target_column or "unset",
        config.task_type,
        summarize_dataset(pipeline_state.dataset),
    )

    try:
        if stage == "analysis":
            from agents.data_analyzer_agent import DataAnalyzerAgent

            agent = DataAnalyzerAgent()
            result = await agent.run(
                pipeline_state.dataset,
                pipeline_state.target_column
            )
            pipeline_state.stage_results["analysis"] = result
            add_agent_summary_logs("analysis", result)
            risk = result.get("risk_level") or "n/a"
            add_log(
                stage,
                f"Analysis complete: {result.get('row_count', 0)} rows, "
                f"{result.get('feature_count', 0)} features, risk={risk}",
            )
            dq = result.get("data_quality") or {}
            quality_parts: list[str] = []
            if dq.get("duplicate_rows", 0):
                quality_parts.append(f"{dq['duplicate_rows']} duplicate rows")
            if dq.get("placeholder_invalid_count", 0):
                quality_parts.append(f"{dq['placeholder_invalid_count']} placeholder-invalid column(s)")
            if dq.get("leakage_risk_columns"):
                quality_parts.append(f"{len(dq['leakage_risk_columns'])} leakage-risk column(s)")
            if quality_parts:
                add_log(stage, "Quality signals: " + ", ".join(quality_parts))

        elif stage == "preprocessing":
            from agents.preprocessor_agent import PreprocessorAgent

            analysis = pipeline_state.stage_results.get("analysis", {})
            agent = PreprocessorAgent()
            result = await agent.run(
                pipeline_state.dataset,
                analysis,
                pipeline_state.target_column,
                test_size=config.test_size,
                random_state=config.random_state,
                config_overrides=pipeline_state.stage_configs.get("preprocessing", {}),
            )
            pipeline_state.stage_results["preprocessing"] = result
            add_agent_summary_logs("preprocessing", result)
            add_log(stage, "Preprocessing complete")

        elif stage == "features":
            from agents.feature_engineering_agent import FeatureEngineeringAgent

            preprocessing = pipeline_state.stage_results.get("preprocessing", {})
            agent = FeatureEngineeringAgent()
            result = await agent.run(
                pipeline_state.dataset,
                preprocessing,
                pipeline_state.target_column,
                config_overrides=pipeline_state.stage_configs.get("feature_engineering", {}),
            )
            pipeline_state.stage_results["features"] = result
            add_agent_summary_logs("features", result)
            add_log(stage, f"Feature engineering complete: {result.get('final_feature_count', 0)} features")

        elif stage == "model_selection":
            from agents.model_selection_agent import ModelSelectionAgent

            analysis = pipeline_state.stage_results.get("analysis", {})
            features = pipeline_state.stage_results.get("features", {})
            model_selection = pipeline_state.stage_results.get("model_selection", {})
            agent = ModelSelectionAgent()
            model_result = await agent.run(
                pipeline_state.dataset,
                features,
                pipeline_state.target_column,
                config.task_type,
                analysis,
            )
            model_result["target_column"] = pipeline_state.target_column
            pipeline_state.stage_results["model_selection"] = model_result
            add_agent_summary_logs("model_selection", model_result)
            if model_result.get("llm_returned"):
                add_log(stage, "Model selection LLM returned a result; using LLM selection.")
            else:
                add_log(stage, "Model selection LLM did not return a result; falling back to default selection.")
            top_candidates = model_result.get("top_candidates", [])
            candidate_names = [item.get("model_name", "unknown") for item in top_candidates if isinstance(item, dict)]
            add_log(stage, f"Candidate set: {', '.join(candidate_names) if candidate_names else 'unknown'}")

        elif stage == "training":
            from agents.training_agent import TrainingAgent

            model_result = pipeline_state.stage_results.get("model_selection")
            if not model_result:
                from agents.model_selection_agent import ModelSelectionAgent

                analysis = pipeline_state.stage_results.get("analysis", {})
                features = pipeline_state.stage_results.get("features", {})
                agent = ModelSelectionAgent()
                model_result = await agent.run(
                    pipeline_state.dataset,
                    features,
                    pipeline_state.target_column,
                    config.task_type,
                    analysis,
                )
                model_result["target_column"] = pipeline_state.target_column
                pipeline_state.stage_results["model_selection"] = model_result
                add_agent_summary_logs("model_selection", model_result)
                if model_result.get("llm_returned"):
                    add_log("model_selection", "Model selection LLM returned a result; using LLM selection.")
                else:
                    add_log("model_selection", "Model selection LLM did not return a result; falling back to default selection.")
                top_candidates = model_result.get("top_candidates", [])
                candidate_names = [item.get("model_name", "unknown") for item in top_candidates if isinstance(item, dict)]
                add_log("model_selection", f"Candidate set: {', '.join(candidate_names) if candidate_names else 'unknown'}")
                pipeline_state.stage_statuses["model_selection"] = "completed"

            train_agent = TrainingAgent()
            training_config = config.model_dump()
            training_overrides = pipeline_state.stage_configs.get("training", {})
            training_config.update(
                {
                    "cv_folds": training_overrides.get("cv_folds") or settings.default_cv_folds,
                    "enable_multi_model": settings.enable_multi_model,
                    "optimize_hyperparameters": (
                        settings.enable_hpo and bool(training_overrides.get("retune_hyperparameters", True))
                    ),
                    "n_trials_hpo": settings.n_hpo_trials,
                    "enable_ensemble": settings.enable_ensemble,
                    "ensemble_type": settings.ensemble_type,
                    "ensemble_top_k": settings.ensemble_top_k,
                    "preprocessing_result": pipeline_state.stage_results.get("preprocessing", {}),
                    "training_overrides": training_overrides,
                }
            )
            train_result = await train_agent.run(
                pipeline_state.dataset,
                model_result,
                training_config,
            )
            pipeline_state.stage_results["training"] = train_result
            add_agent_summary_logs("training", train_result)
            add_log(stage, f"Training complete: {train_result.get('best_score', 0):.4f}")
            try:
                from utils.lightgbm_logger import get_lightgbm_warning_count

                suppressed = get_lightgbm_warning_count()
                if suppressed > 0:
                    add_log(
                        stage,
                        f'LightGBM: suppressed {suppressed} "no further splits with positive gain" warnings.',
                    )
            except Exception:
                pass
            if train_result.get("training_mode") == "multi_model":
                comparisons = train_result.get("model_comparisons", [])
                best_name = train_result.get("model_name", "unknown")
                best_score = train_result.get("best_score", 0)
                add_log(
                    stage,
                    f"Trained {len(comparisons)} candidate models; best = {best_name} (CV={best_score:.4f})",
                )

        elif stage == "loss":
            training = pipeline_state.stage_results.get("training", {})
            loss_source = training.get("loss_source")
            tree_metrics = training.get("tree_metrics")
            if loss_source != "real" and not tree_metrics:
                pipeline_state.stage_results["loss"] = {
                    "train_loss": [],
                    "val_loss": [],
                    "best_epoch": None,
                    "loss_source": loss_source,
                    "tree_metrics": tree_metrics,
                }
                add_log(
                    stage,
                    "Loss curves are only shown for real training histories; "
                    f"current source: {loss_source or 'unavailable'}.",
                )
                pipeline_state.stage_statuses[stage] = "completed"
                add_log(stage, f"{stage} stage completed successfully")
                return
            pipeline_state.stage_results["loss"] = {
                "train_loss": training.get("train_loss", [0.9, 0.6, 0.4, 0.3, 0.2]),
                "val_loss": training.get("val_loss", [0.95, 0.7, 0.5, 0.4, 0.35]),
                "best_epoch": training.get("best_epoch", 3),
                "loss_source": loss_source,
                "tree_metrics": tree_metrics,
            }
            add_log(stage, f"Loss analysis complete. Best epoch: {training.get('best_epoch', 3)}")

        elif stage == "evaluation":
            from agents.evaluation_agent import EvaluationAgent

            training = pipeline_state.stage_results.get("training", {})
            agent = EvaluationAgent()
            result = await agent.run(
                training,
                config.task_type,
                evaluation_config=pipeline_state.stage_configs.get("evaluation", {}),
            )
            llm_insights = generate_evaluation_insights(
                training,
                result,
                target_column=pipeline_state.target_column,
                technical_logs=[
                    *pipeline_state.stage_logs.get("loss", []),
                    *pipeline_state.stage_logs.get("evaluation", []),
                ],
                require_openrouter=True,
            )
            result["llm_insights"] = llm_insights
            result["llm_insights_path"] = persist_evaluation_insights(pipeline_state.pipeline_id, llm_insights)
            pipeline_state.stage_results["evaluation"] = result
            add_agent_summary_logs("evaluation", result)
            add_log(stage, format_evaluation_log(result))
            add_log(stage, "Evaluation OpenRouter insights saved")

        elif stage == "results":
            from agents.deployment_agent import DeploymentAgent
            from agents.explanation_generator_agent import ExplanationGeneratorAgent

            training = pipeline_state.stage_results.get("training", {})
            evaluation = pipeline_state.stage_results.get("evaluation", {})
            analysis = pipeline_state.stage_results.get("analysis", {})
            preprocessing = pipeline_state.stage_results.get("preprocessing", {})
            features = pipeline_state.stage_results.get("features", {})
            model_selection = pipeline_state.stage_results.get("model_selection", {})

            # Run explanation first so README generation has richer context
            explanation_agent = ExplanationGeneratorAgent()
            explanation_result = await explanation_agent.run(
                training,
                evaluation,
                pipeline_context={
                    "dataset": pipeline_state.dataset_filename,
                    "target_column": pipeline_state.target_column,
                    "task_type": config.task_type,
                    "model_selection": pipeline_state.stage_results.get("model_selection", {}),
                    "training": training,
                    "evaluation": evaluation,
                },
            )
            pipeline_state.stage_results["explanation"] = explanation_result
            add_agent_summary_logs("results", explanation_result)
            add_log(stage, "Explanation summary generated")

            agent = DeploymentAgent()
            result = await agent.run(
                training,
                evaluation,
                pipeline_state.pipeline_id,
                dataset_name=pipeline_state.dataset_filename,
                analysis_result=analysis,
                preprocessing_result=preprocessing,
                features_result=features,
                model_selection_result=model_selection,
                explanation_result=explanation_result,
                raw_dataset=pipeline_state.dataset,
                target_column=pipeline_state.target_column,
            )
            pipeline_state.stage_results["results"] = result
            add_agent_summary_logs("results", result)
            add_log(stage, f"Model saved to: {result.get('model_path', 'unknown')}")
            if result.get("package_ready"):
                add_log(stage, "Deployment package ready for download")
            if result.get("report_ready"):
                add_log(stage, "Pipeline report is ready (HTML)")
                
            cleanup_upload()
        
        pipeline_state.stage_statuses[stage] = "completed"
        add_log(stage, f"{stage} stage completed successfully")
        logger.info(
            "Completed stage=%s | result=%s",
            stage,
            json.dumps(summarize_stage_result(stage, pipeline_state.stage_results.get(stage)), default=str, ensure_ascii=True),
        )

    except Exception as e:
        pipeline_state.stage_statuses[stage] = "failed"
        add_log(stage, f"Error: {str(e)}")
        logger.exception("Stage failed stage=%s | error=%s", stage, str(e))
        raise


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "FlowML AutoML API", "version": "1.0.0"}


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file."""
    # Validate file extension
    allowed_extensions = {".csv", ".json", ".xlsx"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )

    # Generate unique ID and save file
    dataset_id = str(uuid.uuid4())
    file_path = UPLOADS_DIR / f"{dataset_id}{file_ext}"

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load into pandas
        if file_ext == ".csv":
            df = pd.read_csv(file_path)
        elif file_ext == ".json":
            df = pd.read_json(file_path)
        elif file_ext == ".xlsx":
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Drop common index-like columns automatically (e.g., 'Unnamed: 0')
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]

        pipeline_state.reset_for_dataset(
            df=df,
            dataset_path=str(file_path),
            dataset_filename=file.filename,
            pipeline_id=dataset_id,
        )
        logger.info(
            "Dataset uploaded | file=%s | pipeline_id=%s | %s",
            file.filename,
            dataset_id,
            summarize_dataset(df),
        )

        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/api/dataset/summary")
async def get_dataset_summary():
    """Get dataset summary metadata."""
    if pipeline_state.dataset is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded")

    df = pipeline_state.dataset

    # Get numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_summary = None
    if numeric_cols:
        numeric_summary = df[numeric_cols].describe().to_dict()

    return {
        "filename": pipeline_state.dataset_filename or "unknown",
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().mean().to_dict(),
        "numeric_summary": numeric_summary,
    }


@app.get("/api/dataset/columns")
async def get_columns():
    """Get column names from the dataset."""
    if pipeline_state.dataset is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded")

    columns = []
    for col in pipeline_state.dataset.columns:
        dtype = str(pipeline_state.dataset[col].dtype)
        is_numeric = np.issubdtype(pipeline_state.dataset[col].dtype, np.number)
        columns.append({
            "name": col,
            "dtype": dtype,
            "is_numeric": is_numeric,
            "missing_pct": float(pipeline_state.dataset[col].isnull().mean()) if col in pipeline_state.dataset.columns else 0,
        })

    return {"columns": columns}


@app.get("/api/dataset/preview")
async def get_dataset_preview(rows: int = 5):
    """Get a small preview of the dataset rows."""
    if pipeline_state.dataset is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded")

    safe_rows = max(1, min(rows, 20))
    df = pipeline_state.dataset.head(safe_rows)
    return {
        "columns": list(df.columns),
        "rows": df.fillna("").to_dict(orient="records"),
    }

@app.post("/api/dataset/target")
async def set_target_column(request: TargetColumnRequest):
    """Set the target column for prediction."""
    if pipeline_state.dataset is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded")

    if request.target_column not in pipeline_state.dataset.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{request.target_column}' not found in dataset"
        )

    pipeline_state.target_column = request.target_column
    add_log("analysis", f"Target column set to: {request.target_column}")

    return {"target_column": request.target_column}


@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Get the current pipeline status."""
    return {
        "pipeline_id": pipeline_state.pipeline_id,
        "stages": pipeline_state.stage_statuses,
        "target_column": pipeline_state.target_column,
        "dataset_loaded": pipeline_state.dataset is not None,
    }


@app.get("/api/pipeline/logs")
async def get_pipeline_logs(stage: Optional[str] = None):
    """Get logs for a specific stage or all stages."""
    if stage:
        if stage not in pipeline_state.stage_logs:
            raise HTTPException(status_code=404, detail=f"Stage '{stage}' not found")
        return {"stage": stage, "logs": pipeline_state.stage_logs[stage]}

    return {"logs": pipeline_state.stage_logs}


@app.post("/api/pipeline/start")
async def start_pipeline(config: PipelineConfig = PipelineConfig()):
    """Start the AutoML pipeline."""
    if pipeline_state.dataset is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded")

    if pipeline_state.target_column is None:
        raise HTTPException(status_code=400, detail="Target column not set")

    pipeline_state.update_pipeline_config(config.model_dump())
    start_new_pipeline_run()

    # Run all stages sequentially
    stages_order = ["analysis", "preprocessing", "features", "model_selection", "training", "loss", "evaluation", "results"]

    for stage in stages_order:
        if pipeline_state.stage_statuses.get(stage) == "waiting":
            await run_pipeline_stage(stage, config)

    maybe_record_revision(
        reason="Initial pipeline run",
        changed_stages=["analysis", "preprocessing", "feature_engineering", "training", "evaluation", "explainability"],
    )

    return {
        "status": "completed",
        "stages": pipeline_state.stage_statuses,
    }


@app.post("/api/pipeline/stage/{stage_id}")
async def run_stage(stage_id: str, config: PipelineConfig = PipelineConfig()):
    """Run a specific pipeline stage."""
    if pipeline_state.dataset is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded")

    if not pipeline_state.pipeline_id:
        start_new_pipeline_run()

    if stage_id not in pipeline_state.stage_statuses:
        raise HTTPException(status_code=404, detail=f"Stage '{stage_id}' not found")

    pipeline_state.update_pipeline_config(config.model_dump())
    await run_pipeline_stage(stage_id, config)

    return {
        "stage_id": stage_id,
        "status": pipeline_state.stage_statuses[stage_id],
        "result": summarize_stage_result(stage_id, pipeline_state.stage_results.get(stage_id)),
    }


@app.get("/api/stages/{stage_id}/results")
async def get_stage_results(stage_id: str):
    """Get results for a specific stage."""
    if stage_id not in pipeline_state.stage_results:
        raise HTTPException(status_code=404, detail=f"No results for stage '{stage_id}'")

    return {
        "stage_id": stage_id,
        "status": pipeline_state.stage_statuses.get(stage_id, "unknown"),
        "result": summarize_stage_result(stage_id, pipeline_state.stage_results[stage_id]),
    }


@app.post("/api/chat/query")
async def query_chat(request: ChatRequest):
    """Answer a user question about the current pipeline run."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    chat_turn_id = f"chat-{uuid.uuid4().hex[:8]}"

    history = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in request.history
        if message.content.strip()
    ]
    selection_context = None
    if request.selection_context:
        selection_context = {
            "text": request.selection_context.text,
            "source_label": request.selection_context.source_label,
            "surrounding_text": request.selection_context.surrounding_text,
        }
    effective_mode = request.mode
    has_pending_revision = bool(pipeline_state.pending_revision_plan)
    should_route_revision = request.mode == "apply" or (
        has_pending_revision and looks_like_revision_request(request.question, request.mode)
    )
    if should_route_revision and has_pending_revision and request.mode != "apply":
        effective_mode = "apply"
    if not should_route_revision:
        preview_plan = chatbot_orchestrator.preview_plan(
            state=pipeline_state,
            question=request.question,
            selection_context=selection_context,
        )
        should_route_revision = (
            preview_plan.intent_type != "other"
            and preview_plan.confidence in {"high", "medium"}
        )
        if should_route_revision and looks_like_revision_request(request.question, request.mode):
            effective_mode = "apply"

    if should_route_revision:
        revision_result = await chatbot_orchestrator.handle_message(
            state=pipeline_state,
            question=request.question,
            mode=effective_mode,
            config=PipelineConfig(**pipeline_state.pipeline_config),
            history=history,
            selection_context=selection_context,
            stage_runner=run_pipeline_stage,
            response_builder=generate_chat_answer,
            request_id=chat_turn_id,
        )
        return ChatResponse(
            answer=revision_result["answer"],
            llm_used=bool(revision_result.get("llm_used", False)),
            response_mode=str(revision_result.get("response_mode") or ("llm" if revision_result.get("llm_used", False) else "structured")),
            revision=revision_result.get("revision"),
        )

    answer, llm_used, response_mode = generate_chat_answer(
        request.question,
        history,
        selection_context,
        request_id=chat_turn_id,
    )
    return ChatResponse(answer=answer, llm_used=llm_used, response_mode=response_mode, revision=None)


@app.get("/api/revisions/current")
async def get_current_revision_state():
    """Return the structured current pipeline state for revision-aware UIs."""
    if pipeline_state.dataset is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded")
    return pipeline_state.current_structured_state()


@app.get("/api/revisions/history")
async def get_revision_history():
    """Return the stored revision history."""
    return {
        "current_run_id": pipeline_state.current_run_id,
        "runs": [record.to_dict() for record in pipeline_state.revision_history],
    }


@app.get("/api/results/download/model")
async def download_model():
    """Download the trained model."""
    results = pipeline_state.stage_results.get("results", {})
    model_path = results.get("model_path")

    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="No model available")

    return FileResponse(
        path=model_path,
        filename="model.pkl",
        media_type="application/octet-stream"
    )


@app.get("/api/results/download/deployment-package")
async def download_deployment_package():
    """Download the complete deployment package zip."""
    results = pipeline_state.stage_results.get("results", {})
    package_path = results.get("package_path")

    if not package_path or not Path(package_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Deployment package not available. Run the full pipeline first.",
        )

    pipeline_id = pipeline_state.pipeline_id or "pipeline"
    filename = f"deployment_package_{pipeline_id[:8]}.zip"
    return FileResponse(
        path=package_path,
        filename=filename,
        media_type="application/zip",
    )


@app.get("/api/results/download/report")
async def download_report():
    """Download the generated pipeline HTML report."""
    results = pipeline_state.stage_results.get("results", {})
    report_path = results.get("report_path")

    if not report_path or not Path(report_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Pipeline report not available. Run the full pipeline first.",
        )

    content = Path(report_path).read_text(encoding="utf-8")
    return HTMLResponse(content=content)


@app.get("/api/results/download/logs")
async def download_logs():
    """Download pipeline logs."""
    logs_data = {
        "pipeline_id": pipeline_state.pipeline_id,
        "target_column": pipeline_state.target_column,
        "stages": pipeline_state.stage_statuses,
        "logs": pipeline_state.stage_logs,
        "results": pipeline_state.stage_results,
    }

    log_file = OUTPUTS_DIR / f"{pipeline_state.pipeline_id}_logs.json"
    with open(log_file, "w") as f:
        json.dump(logs_data, f, indent=2, default=str)

    return FileResponse(
        path=str(log_file),
        filename="pipeline_logs.json",
        media_type="application/json"
    )


@app.get("/api/results/metrics")
async def get_metrics():
    """Get evaluation metrics."""
    evaluation = pipeline_state.stage_results.get("evaluation", {})
    training = pipeline_state.stage_results.get("training", {})

    return {
        "task_type": evaluation.get("task_type", "classification"),
        "accuracy": evaluation.get("accuracy", 0),
        "precision": evaluation.get("precision", 0),
        "recall": evaluation.get("recall", 0),
        "f1": evaluation.get("f1", 0),
        "roc_auc": evaluation.get("roc_auc"),
        "r2": evaluation.get("r2"),
        "mae": evaluation.get("mae"),
        "mse": evaluation.get("mse"),
        "rmse": evaluation.get("rmse"),
        "best_score": training.get("best_score", 0),
        "cv_scores": training.get("cv_scores", []),
        "cv_std": training.get("cv_std"),
        "train_score": training.get("train_score"),
        "test_score": training.get("test_score"),
        "model_name": training.get("model_name"),
        "deployment_decision": evaluation.get("deployment_decision"),
        "performance_summary": evaluation.get("performance_summary"),
        "confusion_matrix": evaluation.get("confusion_matrix", []),
        "baseline_metrics": evaluation.get("baseline_metrics"),
    }


@app.get("/api/results/evaluation-insights", response_model=EvaluationInsightsResponse)
async def get_evaluation_insights():
    """Get the saved structured evaluation insights for the dashboard."""
    evaluation = pipeline_state.stage_results.get("evaluation")

    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation results are not available yet")

    saved_insights = evaluation.get("llm_insights")
    if not isinstance(saved_insights, dict):
        raise HTTPException(status_code=404, detail="Saved evaluation insights are not available for this run")

    return saved_insights


@app.get("/api/results/explanation")
async def get_explanation():
    """Get the pipeline explanation summary."""
    explanation = pipeline_state.stage_results.get("explanation")
    if not explanation:
        raise HTTPException(status_code=404, detail="No explanation available")
    return explanation


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
