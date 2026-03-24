"""Tests for custom exceptions.

This module contains tests for the exception classes defined
in core/exceptions.py.
"""

import pytest

from core.exceptions import (
    AgentExecutionError,
    BlogGenerationError,
    ConfigurationError,
    MemoryError,
    PipelineError,
)


class TestBlogGenerationError:
    """Tests for the BlogGenerationError class."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = BlogGenerationError("Test error")
        assert str(error) == "Test error"
        assert error.details == {}

    def test_error_with_details(self) -> None:
        """Test error with additional details."""
        error = BlogGenerationError("Test error", details={"key": "value"})
        assert str(error) == "Test error"
        assert error.details == {"key": "value"}


class TestAgentExecutionError:
    """Tests for the AgentExecutionError class."""

    def test_error_with_agent_name(self) -> None:
        """Test error with agent name."""
        error = AgentExecutionError(
            "Agent failed",
            agent_name="test_agent",
            details={"input": "test"},
        )
        assert "Agent failed" in str(error)
        assert error.agent_name == "test_agent"
        assert error.details == {"input": "test"}


class TestConfigurationError:
    """Tests for the ConfigurationError class."""

    def test_error_with_config_key(self) -> None:
        """Test error with config key."""
        error = ConfigurationError(
            "Config missing",
            config_key="API_KEY",
        )
        assert "Config missing" in str(error)
        assert error.config_key == "API_KEY"


class TestPipelineError:
    """Tests for the PipelineError class."""

    def test_error_with_stage(self) -> None:
        """Test error with pipeline stage."""
        error = PipelineError(
            "Pipeline failed",
            failed_at_stage="content_writer",
        )
        assert "Pipeline failed" in str(error)
        assert error.failed_at_stage == "content_writer"


class TestMemoryError:
    """Tests for the MemoryError class."""

    def test_basic_error(self) -> None:
        """Test basic memory error."""
        error = MemoryError("Memory operation failed")
        assert str(error) == "Memory operation failed"