"""Tests for chat response behavior and revision intent detection."""

import json
from types import SimpleNamespace

import pytest

from api.main import (
    ChatRequest,
    generate_chat_answer,
    looks_like_revision_request,
    query_chat,
)


def sample_chat_context() -> dict:
    """Build a minimal grounded chat context."""
    return {
        "dataset": {
            "filename": "Fish.csv",
            "rows": 159,
            "columns": 7,
            "column_names": ["Species", "Weight", "Length1", "Length2", "Length3", "Height", "Width"],
            "target_column": "Weight",
            "numeric_columns": ["Weight", "Length1", "Length2", "Length3", "Height", "Width"],
            "categorical_columns": ["Species"],
        },
        "stage_statuses": {
            "analysis": "completed",
            "preprocessing": "completed",
            "features": "completed",
            "model_selection": "completed",
            "training": "completed",
            "evaluation": "completed",
            "results": "completed",
        },
        "stage_results": {
            "features": {
                "final_feature_count": 5,
                "selected_features": ["Length3", "Height", "Width", "Length3__mul__Height", "Length3__div__Height"],
                "generated_features": ["Length3__mul__Height", "Length3__div__Height"],
                "dropped_columns": ["Height__mul__Width"],
                "feature_scores": {
                    "Length3": 0.41,
                    "Height": 0.28,
                    "Width": 0.22,
                    "Length3__mul__Height": 0.16,
                },
                "llm_explanations": {
                    "llmUsed": False,
                    "featureExplanations": {
                        "Height": "It adds direct size information that helps estimate how heavy a fish is.",
                    },
                    "droppedFeatureExplanations": {
                        "Height__mul__Width": "it overlapped too much with stronger numeric features.",
                    },
                },
            },
        },
        "metrics": {
            "task_type": "regression",
            "r2": 0.977,
            "rmse": 57.519,
            "model_name": "LightGBM",
            "deployment_decision": "deploy",
        },
    }


def test_revision_detection_catches_without_feature_requests() -> None:
    """Direct execution phrasing should be treated like an apply request."""
    assert looks_like_revision_request(
        "I would like to run the model without Height as a selected feature",
        "suggest",
    )


def test_revision_detection_does_not_route_explanatory_feature_question() -> None:
    """Normal explanatory questions about the current run should stay in chat mode."""
    assert not looks_like_revision_request(
        "Why did we only use 6 raw features for modeling",
        "suggest",
    )


def test_revision_detection_keeps_explanatory_feature_question_in_chat_mode() -> None:
    """Feature explanation questions should not auto-apply a revision."""
    assert not looks_like_revision_request(
        "Why did Height stay as a selected feature?",
        "suggest",
    )


def test_generate_chat_answer_returns_unavailable_when_llm_is_disabled(monkeypatch) -> None:
    """Normal chat should return an explicit unavailable message instead of a backup answer."""
    monkeypatch.setattr("api.main.build_chat_context", sample_chat_context)
    monkeypatch.setattr("api.main.chat_client.is_enabled", lambda: False)

    answer, llm_used, response_mode = generate_chat_answer(
        "why did height stay",
        history=[],
        selection_context={"text": "Height"},
    )

    assert llm_used is False
    assert response_mode == "unavailable"
    assert "available right now" in answer.lower()


def test_generate_chat_answer_sends_full_grounded_context_to_llm(monkeypatch) -> None:
    """The LLM prompt should include current pipeline context and any revision metadata."""
    monkeypatch.setattr("api.main.build_chat_context", sample_chat_context)
    monkeypatch.setattr("api.main.chat_client.is_enabled", lambda: True)
    captured: dict[str, str] = {}

    def fake_generate_text(*, system_prompt, user_prompt, **kwargs):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return "Grounded answer"

    monkeypatch.setattr("api.main.chat_client.generate_text", fake_generate_text)

    answer, llm_used, response_mode = generate_chat_answer(
        "why did height stay",
        history=[{"role": "assistant", "content": "Earlier context"}],
        selection_context={"text": "Height", "source_label": "Feature engineering insights"},
        extra_context={
            "response_kind": "feature_selection_explanation",
            "feature_name": "Height",
            "importance_score": 0.28,
        },
        request_id="chat-test",
    )

    assert answer == "Grounded answer"
    assert llm_used is True
    assert response_mode == "llm"
    assert "When the user asks why, give the actual reason" in captured["system_prompt"]

    prompt_payload = json.loads(captured["user_prompt"].split("Chat context:\n", 1)[1])
    assert prompt_payload["selection_context"]["text"] == "Height"
    assert prompt_payload["conversation_history"] == ["assistant: Earlier context"]
    assert prompt_payload["pipeline_context"]["stage_results"]["features"]["selected_features"][1] == "Height"
    assert prompt_payload["pipeline_context"]["request_context"]["feature_name"] == "Height"
    assert prompt_payload["pipeline_context"]["request_context"]["response_kind"] == "feature_selection_explanation"


def test_generate_chat_answer_skips_compact_retry_after_transport_failure(monkeypatch) -> None:
    """A socket-style failure should not trigger a second redundant OpenRouter call."""
    monkeypatch.setattr("api.main.build_chat_context", sample_chat_context)
    monkeypatch.setattr("api.main.chat_client.is_enabled", lambda: True)
    call_count = {"count": 0}

    def fake_generate_text(*args, **kwargs):
        call_count["count"] += 1
        raise RuntimeError(
            "OpenRouter connection failed: [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions"
        )

    monkeypatch.setattr("api.main.chat_client.generate_text", fake_generate_text)

    answer, llm_used, response_mode = generate_chat_answer(
        "why did height stay",
        history=[],
        selection_context={"text": "Height"},
        request_id="chat-test",
    )

    assert call_count["count"] == 1
    assert llm_used is False
    assert response_mode == "unavailable"
    assert "couldn't be reached" in answer.lower() or "failed before a response came back" in answer.lower()


@pytest.mark.asyncio
async def test_query_chat_auto_applies_direct_revision_requests(monkeypatch) -> None:
    """Direct revision commands should be forwarded to the orchestrator in apply mode."""
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        "api.main.chatbot_orchestrator.preview_plan",
        lambda **kwargs: SimpleNamespace(intent_type="exclude_feature", confidence="high"),
    )

    async def fake_handle_message(**kwargs):
        captured["mode"] = kwargs["mode"]
        return {
            "answer": "Applied",
            "llm_used": True,
            "response_mode": "llm",
            "revision": {"mode": "apply", "applied": True},
        }

    monkeypatch.setattr("api.main.chatbot_orchestrator.handle_message", fake_handle_message)

    response = await query_chat(
        ChatRequest(
            question="I would like to run the model without Height as a selected feature",
            history=[],
            mode="suggest",
        )
    )

    assert captured["mode"] == "apply"
    assert response.answer == "Applied"
    assert response.revision == {"mode": "apply", "applied": True}


@pytest.mark.asyncio
async def test_query_chat_applies_pending_revision_when_user_says_proceed(monkeypatch) -> None:
    """A short confirmation should execute the pending revision instead of restarting suggestion mode."""
    captured: dict[str, str] = {}
    original_pending_plan = getattr(__import__("api.main", fromlist=["pipeline_state"]).pipeline_state, "pending_revision_plan", None)
    __import__("api.main", fromlist=["pipeline_state"]).pipeline_state.pending_revision_plan = {
        "user_goal": "Run without Height",
        "rerun_from_stage": "preprocessing",
    }

    async def fake_handle_message(**kwargs):
        captured["mode"] = kwargs["mode"]
        return {
            "answer": "Applied pending revision",
            "llm_used": True,
            "response_mode": "llm",
            "revision": {"mode": "apply", "applied": True},
        }

    monkeypatch.setattr("api.main.chatbot_orchestrator.handle_message", fake_handle_message)

    try:
        response = await query_chat(
            ChatRequest(
                question="proceed",
                history=[],
                mode="suggest",
            )
        )
    finally:
        __import__("api.main", fromlist=["pipeline_state"]).pipeline_state.pending_revision_plan = original_pending_plan

    assert captured["mode"] == "apply"
    assert response.answer == "Applied pending revision"
