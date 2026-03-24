"""Base agent class for the AutoML Pipeline.

This module defines the abstract base class that all agents inherit from,
providing common functionality for async execution, logging, and error handling.
"""

from abc import ABC, abstractmethod
import json
import logging
from typing import Any, TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

from core.exceptions import AgentExecutionError
from utils.logger import get_logger
from utils.openrouter_client import OpenRouterClient

if TYPE_CHECKING:
    from langchain.schema import BaseMessage


class BaseAgent(ABC):
    """Abstract base class for all agents in the auto machine learning pipeline.

    This class provides common functionality including async execution,
    logging, and structured error handling. All agents must inherit from
    this class and implement the `execute` method.

    Attributes:
        name: The name of the agent used for logging and identification.
    """

    def __init__(self, name: str) -> None:
        """Initialize the agent.

        Args:
            name: The name of the agent.
        """
        self.name = name
        self._logger = get_logger(name)
        self._llm_client = OpenRouterClient(name)

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute the agent's primary task.

        This method must be implemented by subclasses to define the
        agent's specific behavior.

        Args:
            *args: Positional arguments for the agent.
            **kwargs: Keyword arguments for the agent.

        Returns:
            The processed output (typically a dictionary with results).

        Raises:
            AgentExecutionError: If the agent fails to execute.
        """
        raise NotImplementedError("Subclasses must implement the execute method")

    async def run(self, *args, **kwargs) -> Any:
        """Run the agent with error handling and logging.

        This method wraps the `execute` method with error handling
        and logging to provide consistent behavior across all agents.

        Args:
            *args: Positional arguments to pass to execute.
            **kwargs: Keyword arguments to pass to execute.

        Returns:
            The processed output.

        Raises:
            AgentExecutionError: If the agent fails to execute.
        """
        dataset_summary = self._find_dataset_summary(args, kwargs)
        if dataset_summary:
            self._logger.info(f"Starting {self.name} execution | dataset={dataset_summary}")
        else:
            self._logger.info(f"Starting {self.name} execution")

        input_summary = self._summarize_inputs(args, kwargs)
        if input_summary:
            self._logger.info(f"{self.name} inputs: {input_summary}")
        try:
            result = await self.execute(*args, **kwargs)
            result = self._attach_agent_summary(result, args, kwargs, dataset_summary)
            self._logger.info(f"{self.name} output: {self._summarize_value(result)}")
            self._logger.info(f"Completed {self.name} execution")
            return result
        except AgentExecutionError:
            raise
        except Exception as e:
            self._logger.exception(f"Error in {self.name}: {e}")
            raise AgentExecutionError(
                f"Agent {self.name} failed: {str(e)}",
                agent_name=self.name,
                details={"original_error": str(e)},
            ) from e

    def _build_prompt(self, template: str, **kwargs: str) -> str:
        """Build a prompt from a template with variable substitution.

        Args:
            template: The prompt template string.
            **kwargs: Variables to substitute in the template.

        Returns:
            The formatted prompt string.
        """
        return template.format(**kwargs)

    def _summarize_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        """Create a compact summary of agent inputs for console logging."""
        parts = []

        for index, value in enumerate(args):
            if isinstance(value, pd.DataFrame):
                continue
            parts.append(f"arg{index}={self._summarize_value(value)}")

        for key, value in kwargs.items():
            if isinstance(value, pd.DataFrame):
                continue
            parts.append(f"{key}={self._summarize_value(value)}")

        return "; ".join(parts)

    def _find_dataset_summary(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[str]:
        """Extract a readable dataset summary from agent arguments if present."""
        values = list(args) + list(kwargs.values())

        for value in values:
            if isinstance(value, pd.DataFrame):
                return self._summarize_dataframe(value)

        training_result = next(
            (value for value in values if isinstance(value, dict) and {"X_test", "y_test"} & set(value.keys())),
            None,
        )
        if isinstance(training_result, dict):
            x_test = training_result.get("X_test")
            y_test = training_result.get("y_test")
            if isinstance(x_test, pd.DataFrame):
                return f"test_rows={len(x_test)}, test_cols={len(x_test.columns)}, target_size={len(y_test) if y_test is not None else 'unknown'}"

        return None

    def _summarize_dataframe(self, df: pd.DataFrame) -> str:
        """Create a concise DataFrame summary for console logs."""
        columns_preview = ", ".join(str(column) for column in df.columns[:6])
        if len(df.columns) > 6:
            columns_preview += ", ..."
        return f"rows={len(df)}, cols={len(df.columns)}, columns=[{columns_preview}]"

    def _summarize_value(self, value: Any) -> str:
        """Summarize complex values so agent logs stay readable."""
        if isinstance(value, pd.DataFrame):
            return self._summarize_dataframe(value)
        if isinstance(value, pd.Series):
            return f"Series(len={len(value)}, name={value.name})"
        if isinstance(value, np.ndarray):
            return f"ndarray(shape={value.shape})"
        if isinstance(value, dict):
            compact = {
                str(key): self._compact_value(item)
                for key, item in value.items()
                if not str(key).startswith("_")
            }
            return self._safe_json(compact)
        if isinstance(value, (list, tuple, set)):
            items = [self._compact_value(item) for item in list(value)[:8]]
            if len(value) > 8:
                items.append("...")
            return self._safe_json(items)
        return repr(value)

    def _compact_value(self, value: Any) -> Any:
        """Convert nested values into compact JSON-serializable shapes."""
        if isinstance(value, pd.DataFrame):
            return {
                "type": "DataFrame",
                "rows": len(value),
                "columns": list(value.columns[:8]),
            }
        if isinstance(value, pd.Series):
            preview = value.head(5).tolist()
            return {
                "type": "Series",
                "name": value.name,
                "len": len(value),
                "preview": preview,
            }
        if isinstance(value, np.ndarray):
            preview = value.flatten()[:5].tolist()
            return {
                "type": "ndarray",
                "shape": list(value.shape),
                "preview": preview,
            }
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            visible_items = [(key, item) for key, item in value.items() if not str(key).startswith("_")]
            keys = [key for key, _ in visible_items[:10]]
            compact: dict[str, Any] = {}
            for key in keys:
                compact[str(key)] = self._compact_value(value[key])
            if len(visible_items) > 10:
                compact["..."] = f"{len(visible_items) - 10} more keys"
            return compact
        if isinstance(value, (list, tuple, set)):
            items = [self._compact_value(item) for item in list(value)[:8]]
            if len(value) > 8:
                items.append("...")
            return items
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def _safe_json(self, value: Any) -> str:
        """Serialize compact values for readable logging."""
        return json.dumps(value, default=str, ensure_ascii=True)

    def _attach_agent_summary(
        self,
        result: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        dataset_summary: Optional[str],
    ) -> Any:
        """Attach a structured execution summary to dict-like agent results."""
        if not isinstance(result, dict):
            return result

        summary = self._build_agent_summary(result, args, kwargs, dataset_summary)
        result["_agent_summary"] = summary
        return result

    def _build_agent_summary(
        self,
        result: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        dataset_summary: Optional[str],
    ) -> dict[str, Any]:
        """Build a structured summary for frontend logs."""
        input_payload = {
            "dataset": dataset_summary,
            "inputs": self._summarize_inputs(args, kwargs),
            "result": self._compact_value(result),
        }

        response = self._generate_llm_json(
            system_prompt=(
                "You are summarizing an AutoML pipeline agent step for a sidebar activity log. "
                "Return ONLY valid JSON with keys 'step_summary', 'decisions_made', 'why', and 'overall_summary'. "
                "'step_summary' should be 1 short sentence about what the step did. "
                "'decisions_made' should be a list of 1 to 3 concise bullets. "
                "'why' should be 1 short sentence explaining the rationale. "
                "'overall_summary' should be 1 short sentence suitable for a sidebar log."
            ),
            user_prompt=f"Agent name: {self.name}\nExecution context:\n{self._safe_json(input_payload)}",
            temperature=0.2,
            max_tokens=500,
        )

        if not response:
            return self._build_fallback_agent_summary(result, args, kwargs, dataset_summary)

        decisions = response.get("decisions_made", [])
        if not isinstance(decisions, list):
            decisions = []

        return {
            "agent": self.name,
            "step_summary": str(response.get("step_summary") or f"{self.name} executed."),
            "decisions_made": [str(item).strip() for item in decisions[:3] if str(item).strip()],
            "why": str(response.get("why") or "This step used the available dataset and pipeline context."),
            "overall_summary": str(response.get("overall_summary") or f"{self.name} completed successfully."),
            "llm_used": True,
        }

    def _build_fallback_agent_summary(
        self,
        result: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        dataset_summary: Optional[str],
    ) -> dict[str, Any]:
        """Create a deterministic summary when the LLM is unavailable."""
        input_summary = self._summarize_inputs(args, kwargs)
        decisions = self._extract_decision_bullets(result)
        return {
            "agent": self.name,
            "step_summary": f"{self.name} processed the available pipeline context.",
            "decisions_made": decisions,
            "why": f"It used the current dataset and inputs to produce this stage output{f' ({dataset_summary})' if dataset_summary else ''}.",
            "overall_summary": f"{self.name} completed with {len(result.keys())} reported output fields.",
            "llm_used": False,
            "input_summary": input_summary,
        }

    def _extract_decision_bullets(self, result: dict[str, Any]) -> list[str]:
        """Generate a few concise decision bullets from an agent result."""
        decision_keys = [
            "selected_model",
            "recommendations",
            "risk_level",
            "deployment_decision",
            "model_name",
            "final_feature_count",
            "best_epoch",
            "feature_count",
            "train_size",
            "test_size",
        ]
        bullets: list[str] = []

        for key in decision_keys:
            if key not in result or result.get(key) in (None, "", []):
                continue
            value = result.get(key)
            if isinstance(value, list):
                preview = ", ".join(str(item) for item in value[:2])
                bullets.append(f"{key}: {preview}")
            else:
                bullets.append(f"{key}: {value}")
            if len(bullets) >= 3:
                break

        if not bullets:
            keys = [key for key in result.keys() if not str(key).startswith("_")]
            bullets.append(f"Produced outputs: {', '.join(keys[:3])}")

        return bullets

    def _llm_enabled(self) -> bool:
        """Return whether OpenRouter is configured for this agent."""
        return self._llm_client.is_enabled()

    def _generate_llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """Generate JSON from OpenRouter, logging and falling back on failure."""
        if not self._llm_enabled():
            return None

        try:
            return self._llm_client.generate_json(
                system_prompt,
                user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            self._logger.warning(f"{self.name} LLM JSON generation failed, using fallback logic: {exc}")
            return None

    def _generate_llm_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Generate text from OpenRouter, logging and falling back on failure."""
        if not self._llm_enabled():
            return None

        try:
            return self._llm_client.generate_text(
                system_prompt,
                user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            self._logger.warning(f"{self.name} LLM text generation failed, using fallback logic: {exc}")
            return None
