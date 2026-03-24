"""Custom exceptions for the AutoML Pipeline application.

This module defines exception classes used throughout the application
for consistent error handling and reporting.
"""


from typing import Optional, Dict


class AutoMLError(Exception):
    """Base exception for all AutoML errors.

    This is the root exception class from which all other application
    exceptions inherit.
    """

    def __init__(self, message: str, details: Optional[Dict] = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message)
        self.details = details or {}


class AgentExecutionError(AutoMLError):
    """Exception raised when an agent fails to execute.

    This exception is raised when an agent encounters an error during
    execution, such as API failures, invalid inputs, or timeouts.
    """

    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            agent_name: Name of the agent that failed.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message, details)
        self.agent_name = agent_name


class ConfigurationError(AutoMLError):
    """Exception raised when configuration is invalid or missing.

    This exception is raised when required configuration values
    are missing or invalid.
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            config_key: The configuration key that caused the error.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message, details)
        self.config_key = config_key


class MemoryError(AutoMLError):
    """Exception raised when memory operations fail.

    This exception is raised when there are issues with memory
    management operations.
    """


class PipelineError(AutoMLError):
    """Exception raised when the agent pipeline fails.

    This exception is raised when the orchestration pipeline
    encounters an error during execution.
    """

    def __init__(
        self,
        message: str,
        failed_at_stage: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            failed_at_stage: The stage at which the pipeline failed.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message, details)
        self.failed_at_stage = failed_at_stage
