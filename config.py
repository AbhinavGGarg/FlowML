"""Configuration management for the AutoML Pipeline application.

This module provides centralized configuration using Pydantic Settings
for environment variable validation and type safety.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application settings
    app_name: str = Field(default="FlowML AutoML", description="Application name")
    app_env: str = Field(default="development", description="Environment: development, staging, production")
    log_level: str = Field(default="INFO", description="Logging level")

    # OpenRouter API settings (optional for local processing)
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key for LLM calls")
    model_name: str = Field(default="arcee-ai/trinity-large-preview:free", description="LLM model to use")
    chat_model_name: Optional[str] = Field(default=None, description="Optional chat-only LLM model override")
    model_fallbacks: str = Field(
        default="",
        description="Comma-separated fallback OpenRouter models to try if the primary model fails",
    )
    model_temperature: float = Field(default=0.7, description="LLM temperature (0.0-1.0)")
    model_max_tokens: int = Field(default=2000, description="Maximum tokens for LLM response")

    # Production settings
    request_timeout: int = Field(default=60, description="HTTP request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests")

    # Pipeline settings
    default_test_size: float = Field(default=0.2, description="Default test set proportion")
    default_random_state: int = Field(default=42, description="Default random seed")
    default_cv_folds: int = Field(default=5, description="Default cross-validation folds")

    # Advanced AutoML settings
    enable_multi_model: bool = Field(default=True, description="Enable multi-model comparison")
    enable_hpo: bool = Field(default=True, description="Enable hyperparameter optimization with Optuna")
    enable_ensemble: bool = Field(default=False, description="Enable ensemble building")
    ensemble_type: str = Field(default="stacking", description="Ensemble type: voting or stacking")
    ensemble_top_k: int = Field(default=3, description="Number of top models for ensemble")
    n_hpo_trials: int = Field(default=20, description="Number of Optuna HPO trials")
    parallel_training: bool = Field(default=True, description="Enable parallel model training")


settings = Settings()


def get_openrouter_api_key() -> str:
    """Retrieve the OpenRouter API key.

    Returns:
        The API key string.

    Raises:
        ValueError: If API key is not configured.
    """
    api_key = settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenRouter API key not configured. Set OPENROUTER_API_KEY in .env file."
        )
    return api_key


def get_openrouter_model_candidates() -> list[str]:
    """Return the ordered list of OpenRouter models to try."""
    raw_candidates = [settings.model_name, *settings.model_fallbacks.split(",")]
    candidates: list[str] = []

    for item in raw_candidates:
        model_name = (item or "").strip()
        if not model_name or model_name in candidates:
            continue
        candidates.append(model_name)

    return candidates
