"""Tests for OpenRouter retry and model failover behavior."""

from types import SimpleNamespace

from config import get_openrouter_model_candidates
from utils.openrouter_client import OpenRouterClient


def test_get_openrouter_model_candidates_dedupes_and_preserves_order(monkeypatch) -> None:
    """Configured fallback models should be parsed into a clean ordered list."""
    mock_settings = SimpleNamespace(
        model_name="primary/model",
        model_fallbacks="backup/model, primary/model, second/model, ",
    )
    monkeypatch.setattr("config.settings", mock_settings)

    assert get_openrouter_model_candidates() == [
        "primary/model",
        "backup/model",
        "second/model",
    ]


def test_generate_text_retries_and_falls_back_to_second_model(monkeypatch) -> None:
    """The client should retry transient errors and then fail over to another LLM model."""
    client = OpenRouterClient("Test")
    attempts: list[tuple[str, int]] = []
    state = {"primary_calls": 0}

    monkeypatch.setattr("utils.openrouter_client.get_openrouter_api_key", lambda: "test-key")
    monkeypatch.setattr(
        "utils.openrouter_client.get_openrouter_model_candidates",
        lambda: ["primary/model", "backup/model"],
    )
    monkeypatch.setattr("utils.openrouter_client.settings.max_retries", 2)
    monkeypatch.setattr("utils.openrouter_client.settings.model_temperature", 0.7)
    monkeypatch.setattr("utils.openrouter_client.settings.model_max_tokens", 2000)
    monkeypatch.setattr("utils.openrouter_client.settings.app_name", "FlowML")
    monkeypatch.setattr("utils.openrouter_client.settings.request_timeout", 5)
    monkeypatch.setattr("utils.openrouter_client.time.sleep", lambda _seconds: None)

    def fake_request_completion(**kwargs):
        model_name = kwargs["model_name"]
        if model_name == "primary/model":
            state["primary_calls"] += 1
            attempts.append((model_name, state["primary_calls"]))
            raise RuntimeError("OpenRouter HTTP error 503: provider overloaded")

        attempts.append((model_name, 1))
        return '{"choices":[{"message":{"content":"OK from backup"}}]}'

    monkeypatch.setattr(client, "_request_completion", fake_request_completion)

    answer = client.generate_text("system", "user")

    assert answer == "OK from backup"
    assert attempts == [
        ("primary/model", 1),
        ("primary/model", 2),
        ("backup/model", 1),
    ]


def test_generate_text_reports_models_tried_when_all_fail(monkeypatch) -> None:
    """The final error should mention which LLM models were attempted."""
    client = OpenRouterClient("Test")

    monkeypatch.setattr("utils.openrouter_client.get_openrouter_api_key", lambda: "test-key")
    monkeypatch.setattr(
        "utils.openrouter_client.get_openrouter_model_candidates",
        lambda: ["primary/model", "backup/model"],
    )
    monkeypatch.setattr("utils.openrouter_client.settings.max_retries", 1)
    monkeypatch.setattr("utils.openrouter_client.settings.model_temperature", 0.7)
    monkeypatch.setattr("utils.openrouter_client.settings.model_max_tokens", 2000)
    monkeypatch.setattr("utils.openrouter_client.settings.app_name", "FlowML")
    monkeypatch.setattr("utils.openrouter_client.settings.request_timeout", 5)

    def fake_request_completion(**kwargs):
        raise RuntimeError(f"OpenRouter connection failed: {kwargs['model_name']}")

    monkeypatch.setattr(client, "_request_completion", fake_request_completion)

    try:
        client.generate_text("system", "user")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected generate_text to raise RuntimeError")

    assert "primary/model" in message
    assert "backup/model" in message


def test_shared_http_client_is_reused_and_can_be_closed() -> None:
    """OpenRouter should reuse one pooled HTTP client instead of creating a new transport each time."""
    OpenRouterClient.close_shared_client()

    first = OpenRouterClient._get_shared_client()
    second = OpenRouterClient._get_shared_client()

    assert first is second

    OpenRouterClient.close_shared_client()
    third = OpenRouterClient._get_shared_client()

    assert third is not first
    OpenRouterClient.close_shared_client()


def test_client_model_override_is_used_before_global_candidates(monkeypatch) -> None:
    """A chat-specific model override should take precedence for that client only."""
    monkeypatch.setattr(
        "utils.openrouter_client.get_openrouter_model_candidates",
        lambda: ["global/model", "fallback/model"],
    )

    client = OpenRouterClient("Chat", model_name="chat/model")

    assert client._get_model_candidates() == ["chat/model"]
