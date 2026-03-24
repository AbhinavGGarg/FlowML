"""Tests for the configuration module.

This module contains tests for the Settings class and
configuration loading.
"""

import os
import pytest
from unittest.mock import patch

from config import Settings, get_openrouter_api_key


class TestSettings:
    """Tests for the Settings class."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = Settings()
        assert settings.app_name == "Blog Generator"
        assert settings.app_env == "development"
        assert settings.log_level == "INFO"

    def test_custom_settings_from_env(self) -> None:
        """Test loading settings from environment variables."""
        with patch.dict(
            os.environ,
            {
                "APP_NAME": "Test App",
                "APP_ENV": "production",
                "LOG_LEVEL": "DEBUG",
                "MODEL_NAME": "test/model",
                "OPENROUTER_API_KEY": "test-key",
            },
        ):
            settings = Settings()
            assert settings.app_name == "Test App"
            assert settings.app_env == "production"
            assert settings.log_level == "DEBUG"
            assert settings.model_name == "test/model"
            assert settings.openrouter_api_key == "test-key"

    def test_model_settings_defaults(self) -> None:
        """Test default model settings."""
        settings = Settings()
        assert settings.model_name == "anthropic/claude-3-haiku"
        assert settings.model_temperature == 0.7
        assert settings.model_max_tokens == 2000

    def test_production_settings_defaults(self) -> None:
        """Test default production settings."""
        settings = Settings()
        assert settings.request_timeout == 60
        assert settings.max_retries == 3


class TestGetOpenRouterApiKey:
    """Tests for the get_openrouter_api_key function."""

    def test_returns_api_key_from_settings(self) -> None:
        """Test returning API key from settings."""
        with patch("config.settings") as mock_settings:
            mock_settings.openrouter_api_key = "test-key"
            result = get_openrouter_api_key()
            assert result == "test-key"

    def test_returns_api_key_from_env(self) -> None:
        """Test returning API key from environment variable."""
        with patch("config.settings") as mock_settings:
            mock_settings.openrouter_api_key = None
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}):
                result = get_openrouter_api_key()
                assert result == "env-key"

    def test_raises_when_no_api_key(self) -> None:
        """Test ValueError when no API key is configured."""
        with patch("config.settings") as mock_settings:
            mock_settings.openrouter_api_key = None
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError) as exc_info:
                    get_openrouter_api_key()
                assert "OpenRouter API key not configured" in str(exc_info.value)