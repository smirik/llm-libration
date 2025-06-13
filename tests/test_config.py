"""Tests for the configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from llm_libration.config import Config
from llm_libration.exceptions import ConfigurationError


class TestConfig:
    """Test cases for Config class."""

    def test_config_loads_environment_variables(self):
        """Test that config loads environment variables correctly."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key", "OPENAI_MODEL_NAME": "gpt-4"}):
            config = Config(load_env=False)
            assert config.llm_provider == "openai"
            assert config.openai_api_key == "test-key"
            assert config.openai_model_name == "gpt-4"

    def test_config_default_values(self):
        """Test that config uses default values when env vars are not set."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            # Mock Path to prevent loading any .env files for this test
            with patch('llm_libration.config.Path') as mock_path_class:
                mock_path = mock_path_class.return_value
                mock_path.exists.return_value = False  # No .env files exist

                config = Config(load_env=False)
                assert config.llm_provider == "openai"  # Default provider
                assert config.openai_api_key == "test-key"
                assert config.openai_model_name == "gpt-4.1-mini-2025-04-14"
                assert len(config.prompt_template) > 100

    def test_config_validation_openai_success(self):
        """Test successful validation for OpenAI provider."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}):
            config = Config()
            config.validate_required_env_vars()  # Should not raise

    def test_config_validation_openai_failure(self):
        """Test validation failure when OpenAI API key is missing."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            config = Config(load_env=False)
            with pytest.raises(ConfigurationError, match="OPENAI_API_KEY not found"):
                config.validate_required_env_vars()

    def test_config_validation_anthropic_success(self):
        """Test successful validation for Anthropic provider."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"}):
            config = Config()
            config.validate_required_env_vars()  # Should not raise

    def test_config_validation_anthropic_failure(self):
        """Test validation failure when Anthropic API key is missing."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}, clear=True):
            config = Config(load_env=False)
            with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY not found"):
                config.validate_required_env_vars()

    def test_config_validation_openrouter_success(self):
        """Test successful validation for OpenRouter provider."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "test-key"}):
            config = Config()
            config.validate_required_env_vars()  # Should not raise

    def test_config_validation_openrouter_failure(self):
        """Test validation failure when OpenRouter API key is missing."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}, clear=True):
            config = Config(load_env=False)
            with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY not found"):
                config.validate_required_env_vars()

    def test_config_validation_ollama_success(self):
        """Test successful validation for Ollama provider."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "OLLAMA_BASE_URL": "http://localhost:11434"}):
            config = Config()
            config.validate_required_env_vars()  # Should not raise

    def test_config_validation_ollama_failure(self):
        """Test validation failure when Ollama base URL is missing."""
        # Mock both the environment and the .env file loading
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=True):
            # Mock the file loading to prevent .env.dist from being loaded
            with patch('llm_libration.config.Path') as mock_path_class:
                mock_path = mock_path_class.return_value
                mock_path.exists.return_value = False  # No .env files exist

                config = Config()
                with pytest.raises(ConfigurationError, match="OLLAMA_BASE_URL not found"):
                    config.validate_required_env_vars()

    def test_config_invalid_provider(self):
        """Test validation failure with invalid provider."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "invalid_provider"}):
            config = Config(load_env=False)
            with pytest.raises(ConfigurationError, match="Invalid LLM_PROVIDER"):
                config.llm_provider

    def test_config_unsupported_provider_validation(self):
        """Test validation failure with unsupported provider."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "unsupported"}):
            config = Config(load_env=False)
            with pytest.raises(ConfigurationError, match="Invalid LLM_PROVIDER"):
                config.llm_provider

    def test_config_custom_prompt_template(self):
        """Test that custom prompt template is loaded from environment."""
        custom_prompt = "This is a custom prompt template"
        with patch.dict(os.environ, {"PROMPT_TEMPLATE": custom_prompt, "OPENAI_API_KEY": "test-key"}):
            config = Config(load_env=False)
            assert config.prompt_template == custom_prompt

    def test_config_anthropic_properties(self):
        """Test Anthropic-specific configuration properties."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key", "ANTHROPIC_MODEL_NAME": "claude-sonnet-4"}):
            config = Config(load_env=False)
            assert config.anthropic_api_key == "test-anthropic-key"
            assert config.anthropic_model_name == "claude-sonnet-4"

    def test_config_openrouter_properties(self):
        """Test OpenRouter-specific configuration properties."""
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-openrouter-key",
                "OPENROUTER_MODEL_NAME": "anthropic/claude-sonnet-4",
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
            },
        ):
            config = Config(load_env=False)
            assert config.openrouter_api_key == "test-openrouter-key"
            assert config.openrouter_model_name == "anthropic/claude-sonnet-4"
            assert config.openrouter_base_url == "https://openrouter.ai/api/v1"

    def test_config_ollama_properties(self):
        """Test Ollama-specific configuration properties."""
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434", "OLLAMA_MODEL_NAME": "gemma3"}):
            config = Config(load_env=False)
            assert config.ollama_base_url == "http://localhost:11434"
            assert config.ollama_model_name == "gemma3"

    def test_config_default_model_name_by_provider(self):
        """Test that default_model_name returns correct model based on provider."""
        # Test OpenAI
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_MODEL_NAME": "gpt-4.1-2025-04-14"}):
            config = Config(load_env=False)
            assert config.default_model_name == "gpt-4.1-2025-04-14"

        # Test Anthropic
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_MODEL_NAME": "claude-sonnet-4"}):
            config = Config(load_env=False)
            assert config.default_model_name == "claude-sonnet-4"

        # Test OpenRouter
        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter", "OPENROUTER_MODEL_NAME": "anthropic/claude-sonnet-4"}):
            config = Config(load_env=False)
            assert config.default_model_name == "anthropic/claude-sonnet-4"

        # Test Ollama
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL_NAME": "gemma3"}):
            config = Config(load_env=False)
            assert config.default_model_name == "gemma3"

    def test_config_loads_env_dist_then_env(self):
        """Test that config loads .env.dist first, then .env with override."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary .env.dist file
            env_dist_path = Path(temp_dir) / ".env.dist"
            with open(env_dist_path, "w") as f:
                f.write("LLM_PROVIDER=openai\n")
                f.write("OPENAI_API_KEY=from-env-dist\n")
                f.write("OPENAI_MODEL_NAME=model-from-dist\n")

            # Create temporary .env file
            env_path = Path(temp_dir) / ".env"
            with open(env_path, "w") as f:
                f.write("OPENAI_API_KEY=from-env\n")
                # Note: not overriding OPENAI_MODEL_NAME

            # Mock Path.cwd() to return our temp directory
            with patch('llm_libration.config.Path') as mock_path_class:
                # Mock the Path(".env.dist") and Path(".env") calls
                def mock_path_constructor(path_str):
                    if path_str == ".env.dist":
                        return env_dist_path
                    elif path_str == ".env":
                        return env_path
                    else:
                        return Path(path_str)

                mock_path_class.side_effect = mock_path_constructor

                # Clear environment to ensure we're testing file loading
                with patch.dict(os.environ, {}, clear=True):
                    config = Config()

                    # Should have values from both files, with .env overriding .env.dist
                    assert config.openai_api_key == "from-env"  # Overridden
                    assert config.openai_model_name == "model-from-dist"  # From .env.dist

    def test_config_env_file_not_exists(self):
        """Test that config works when .env files don't exist."""
        with patch('llm_libration.config.Path') as mock_path_class:
            # Mock Path to return non-existent files
            mock_path = mock_path_class.return_value
            mock_path.exists.return_value = False

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                config = Config()
                # Should work with system environment variables
                assert config.openai_api_key == "test-key"

    def test_prompt_template_contains_required_keywords(self):
        """Test that the default prompt template contains required keywords."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = Config()
            prompt = config.prompt_template.lower()

            # Check for required keywords
            assert "astronomer" in prompt
            assert "pure" in prompt
            assert "transient" in prompt
            assert "non-resonant" in prompt
            assert "librat" in prompt  # libration/librates
            assert "resonant angle" in prompt

    def test_global_config_instance(self):
        """Test that the global config instance works correctly."""
        from llm_libration.config import config

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Should be able to access properties
            assert config.openai_api_key == "test-key"
            assert isinstance(config.default_model_name, str)
            assert isinstance(config.prompt_template, str)

    def test_config_load_env_parameter(self):
        """Test that the load_env parameter controls .env file loading."""
        # Test with load_env=True (default behavior)
        with patch.dict(os.environ, {"TEST_VAR": "from_env"}, clear=True):
            config_with_env = Config(load_env=True)
            # This test mainly verifies the parameter exists and doesn't crash
            assert isinstance(config_with_env.llm_provider, str)

        # Test with load_env=False
        with patch.dict(os.environ, {"TEST_VAR": "from_env", "LLM_PROVIDER": "openai"}, clear=True):
            config_without_env = Config(load_env=False)
            # Should still work with environment variables, just not load .env file
            assert config_without_env.llm_provider == "openai"
