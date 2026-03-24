"""Minimal OpenRouter client for optional agentic reasoning."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from typing import Any, Optional

import httpx

from config import get_openrouter_api_key, get_openrouter_model_candidates, settings
from utils.logger import get_logger


class OpenRouterClient:
    """Small OpenRouter client with graceful fallback behavior."""

    _shared_client: Optional[httpx.Client] = None
    _client_lock = threading.Lock()

    def __init__(
        self,
        logger_name: str = "OpenRouter",
        *,
        model_name: Optional[str] = None,
        model_fallbacks: Optional[list[str]] = None,
    ) -> None:
        self._logger = get_logger(f"{logger_name}.OpenRouter")
        self._model_name_override = model_name.strip() if isinstance(model_name, str) and model_name.strip() else None
        self._model_fallbacks_override = [
            item.strip() for item in (model_fallbacks or [])
            if isinstance(item, str) and item.strip()
        ]

    def is_enabled(self) -> bool:
        """Return whether OpenRouter is configured."""
        return bool(settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY"))

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Generate a plain-text response from OpenRouter."""
        api_key = get_openrouter_api_key()
        last_error: Optional[RuntimeError] = None
        models_tried: list[str] = []
        request_id = request_id or uuid.uuid4().hex[:12]

        for model_name in self._get_model_candidates():
            models_tried.append(model_name)
            for attempt in range(1, max(settings.max_retries, 1) + 1):
                try:
                    body = self._request_completion(
                        api_key=api_key,
                        model_name=model_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        request_id=request_id,
                        attempt=attempt,
                    )
                    payload = json.loads(body)
                    return self._extract_text_from_payload(payload)
                except RuntimeError as exc:
                    last_error = exc
                    if self._should_retry(exc, attempt):
                        self._logger.warning(
                            "OpenRouter attempt %s/%s failed for model %s: %s",
                            attempt,
                            max(settings.max_retries, 1),
                            model_name,
                            exc,
                        )
                        time.sleep(self._retry_delay_seconds(attempt))
                        continue

                    self._logger.warning(
                        "OpenRouter failed for model %s on attempt %s/%s: %s",
                        model_name,
                        attempt,
                        max(settings.max_retries, 1),
                        exc,
                    )
                    break

        attempted = ", ".join(models_tried) or settings.model_name
        detail = str(last_error) if last_error else "unknown error"
        raise RuntimeError(f"OpenRouter failed for models [{attempted}]. Last error: {detail}")

    def _get_model_candidates(self) -> list[str]:
        """Return model candidates for this client, allowing per-client overrides."""
        if not self._model_name_override and not self._model_fallbacks_override:
            return get_openrouter_model_candidates()

        raw_candidates = [self._model_name_override or settings.model_name, *self._model_fallbacks_override]
        candidates: list[str] = []
        for item in raw_candidates:
            model_name = (item or "").strip()
            if not model_name or model_name in candidates:
                continue
            candidates.append(model_name)
        return candidates

    def _request_completion(
        self,
        *,
        api_key: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        request_id: str,
        attempt: int,
    ) -> str:
        """Execute a single OpenRouter completion request."""
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": settings.model_temperature if temperature is None else temperature,
            "max_tokens": settings.model_max_tokens if max_tokens is None else max_tokens,
        }
        client = self._get_shared_client()
        started = time.perf_counter()
        self._logger.info(
            "OPENROUTER CALL START request_id=%s model=%s attempt=%s",
            request_id,
            model_name,
            attempt,
        )

        try:
            response = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": settings.app_name,
                },
            )
            response.raise_for_status()
            duration_ms = (time.perf_counter() - started) * 1000
            self._logger.info(
                "OPENROUTER CALL END request_id=%s model=%s attempt=%s status=%s duration_ms=%.1f",
                request_id,
                model_name,
                attempt,
                response.status_code,
                duration_ms,
            )
            return response.text
        except httpx.HTTPStatusError as exc:
            duration_ms = (time.perf_counter() - started) * 1000
            detail = exc.response.text
            self._logger.warning(
                "OPENROUTER CALL FAIL request_id=%s model=%s attempt=%s status=%s duration_ms=%.1f error=%s",
                request_id,
                model_name,
                attempt,
                exc.response.status_code,
                duration_ms,
                detail,
            )
            raise RuntimeError(f"OpenRouter HTTP error {exc.response.status_code}: {detail}") from exc
        except httpx.TimeoutException as exc:
            duration_ms = (time.perf_counter() - started) * 1000
            self._logger.warning(
                "OPENROUTER CALL FAIL request_id=%s model=%s attempt=%s duration_ms=%.1f error=%s",
                request_id,
                model_name,
                attempt,
                duration_ms,
                exc,
            )
            raise RuntimeError("OpenRouter request timed out") from exc
        except httpx.NetworkError as exc:
            duration_ms = (time.perf_counter() - started) * 1000
            self._logger.warning(
                "OPENROUTER CALL FAIL request_id=%s model=%s attempt=%s duration_ms=%.1f error=%s",
                request_id,
                model_name,
                attempt,
                duration_ms,
                exc,
            )
            raise RuntimeError(f"OpenRouter connection failed: {exc}") from exc
        except OSError as exc:
            duration_ms = (time.perf_counter() - started) * 1000
            self._logger.warning(
                "OPENROUTER CALL FAIL request_id=%s model=%s attempt=%s duration_ms=%.1f error=%s",
                request_id,
                model_name,
                attempt,
                duration_ms,
                exc,
            )
            raise RuntimeError(f"OpenRouter socket error: {exc.strerror or exc}") from exc

    def _extract_text_from_payload(self, payload: dict[str, Any]) -> str:
        """Normalize the response payload into a plain string."""
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("OpenRouter returned no choices")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                    continue
                if isinstance(part.get("content"), str):
                    text_parts.append(part["content"])
                    continue
                if isinstance(part.get("value"), str):
                    text_parts.append(part["value"])
                    continue
                if isinstance(part.get("type"), str) and part.get("type") == "text" and isinstance(part.get("data"), str):
                    text_parts.append(part["data"])
            assembled = "\n".join(part for part in text_parts if part).strip()
            if assembled:
                return assembled
            self._logger.warning("OpenRouter returned an empty content parts payload: %s", json.dumps(payload, ensure_ascii=True))
            raise RuntimeError("OpenRouter returned an unsupported content payload")

        if isinstance(content, str):
            return content.strip()

        if content is None and isinstance(message.get("reasoning"), str):
            return message["reasoning"].strip()

        self._logger.warning("OpenRouter returned an unsupported content payload: %s", json.dumps(payload, ensure_ascii=True))
        raise RuntimeError("OpenRouter returned an unsupported content payload")

    def _should_retry(self, error: RuntimeError, attempt: int) -> bool:
        """Return whether a request error is worth retrying."""
        if attempt >= max(settings.max_retries, 1):
            return False

        message = str(error).lower()
        retryable_markers = (
            "timed out",
            "timeout",
            "connection reset",
            "temporarily unavailable",
            "bad gateway",
            "gateway timeout",
            "service unavailable",
            "too many requests",
            "http error 408",
            "http error 409",
            "http error 425",
            "http error 429",
            "http error 500",
            "http error 502",
            "http error 503",
            "http error 504",
        )
        if any(marker in message for marker in retryable_markers):
            return True

        socket_markers = (
            "connection failed",
            "socket error",
            "network error",
            "connecterror",
            "readerror",
            "name or service not known",
            "nodename nor servname provided",
            "timed out",
        )
        if any(marker in message for marker in socket_markers):
            return "forbidden by its access permissions" not in message

        return False

    def _retry_delay_seconds(self, attempt: int) -> float:
        """Return a small bounded backoff delay."""
        return min(0.5 * (2 ** (attempt - 1)), 2.0)

    @classmethod
    def _get_shared_client(cls) -> httpx.Client:
        """Return a process-wide pooled HTTP client for OpenRouter."""
        with cls._client_lock:
            if cls._shared_client is None or cls._shared_client.is_closed:
                timeout = httpx.Timeout(
                    connect=min(float(settings.request_timeout), 10.0),
                    read=float(settings.request_timeout),
                    write=float(settings.request_timeout),
                    pool=5.0,
                )
                limits = httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                    keepalive_expiry=30.0,
                )
                cls._shared_client = httpx.Client(
                    timeout=timeout,
                    limits=limits,
                    http2=False,
                )
            return cls._shared_client

    @classmethod
    def close_shared_client(cls) -> None:
        """Close the shared pooled HTTP client."""
        with cls._client_lock:
            if cls._shared_client is not None and not cls._shared_client.is_closed:
                cls._shared_client.close()
            cls._shared_client = None

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate a JSON object from OpenRouter."""
        response_text = self.generate_text(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            request_id=request_id,
        )
        try:
            return self._extract_json_object(response_text)
        except Exception as exc:
            self._logger.warning(
                "OpenRouter JSON parse failed. Raw response (truncated): %s",
                self._truncate_for_log(response_text),
            )
            raise RuntimeError(str(exc)) from exc

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        """Extract a JSON object from a model response."""
        stripped = self._normalize_json_text(text)
        candidates: list[str] = []

        candidates.append(stripped)
        fence_stripped = self._strip_code_fences(stripped)
        if fence_stripped != stripped:
            candidates.append(fence_stripped)

        balanced_object = self._find_balanced_object(stripped)
        if balanced_object:
            candidates.append(balanced_object)

        if fence_stripped != stripped:
            fenced_balanced = self._find_balanced_object(fence_stripped)
            if fenced_balanced:
                candidates.append(fenced_balanced)

        parse_errors: list[str] = []
        for candidate in self._dedupe_candidates(candidates):
            payload = self._try_parse_json_object(candidate, parse_errors)
            if payload is not None:
                return payload

            repaired = self._repair_common_json_issues(candidate)
            if repaired != candidate:
                payload = self._try_parse_json_object(repaired, parse_errors)
                if payload is not None:
                    return payload

        if parse_errors:
            raise RuntimeError(f"OpenRouter returned malformed JSON: {parse_errors[-1]}")

        raise RuntimeError(f"OpenRouter did not return JSON: {stripped}")

    def _normalize_json_text(self, text: str) -> str:
        """Normalize common model response wrappers before JSON extraction."""
        normalized = text.lstrip("\ufeff").strip()

        # Handle responses like: "json\n{...}" or "JSON: {...}"
        lowered = normalized.lower()
        if lowered.startswith("json"):
            remainder = normalized[4:]
            normalized = remainder.lstrip(" :\n\r\t")

        # Handle opening fence even if closing fence is missing.
        if normalized.startswith("```"):
            lines = normalized.splitlines()
            if lines:
                normalized = "\n".join(lines[1:]).strip()

        return normalized

    def _try_parse_json_object(
        self,
        text: str,
        parse_errors: list[str],
    ) -> Optional[dict[str, Any]]:
        """Attempt to parse text as a JSON object."""
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
            parse_errors.append("JSON payload was not an object")
            return None
        except json.JSONDecodeError as exc:
            parse_errors.append(str(exc))
            return None

    def _strip_code_fences(self, text: str) -> str:
        """Strip markdown code fences if present."""
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if len(lines) < 3:
            return stripped

        if lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()

        # If closing fence is missing, still remove the opening fence line.
        if lines[0].startswith("```"):
            return "\n".join(lines[1:]).strip()

        return stripped

    def _find_balanced_object(self, text: str) -> Optional[str]:
        """Find the first balanced JSON object in text."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for index, char in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]

        return None

    def _repair_common_json_issues(self, text: str) -> str:
        """Repair common model JSON formatting issues deterministically."""
        repaired = text
        repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
        repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(r"\n\s*```.*$", "", repaired, flags=re.MULTILINE)
        return repaired.strip()

    def _dedupe_candidates(self, candidates: list[str]) -> list[str]:
        """Return candidates preserving order and removing duplicates."""
        seen: set[str] = set()
        unique: list[str] = []
        for item in candidates:
            if not item or item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

    def _truncate_for_log(self, text: str, max_chars: int = 4000) -> str:
        """Trim long LLM payloads for readable logs."""
        normalized = text.replace("\n", "\\n")
        if len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars] + "... [truncated]"
