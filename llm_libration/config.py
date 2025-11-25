"""Configuration management for the LLM Libration project."""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from .exceptions import ConfigurationError

# Supported LLM providers
LLMProvider = Literal["openai", "anthropic", "openrouter", "ollama"]


class Config:
    """Centralized configuration management for environment variables."""

    def __init__(self, load_env: bool = True):
        """Initialize configuration by loading environment variables.

        Args:
            load_env: Whether to load .env file. If False, only .env.dist is loaded.
                     Defaults to True for backward compatibility.
        """
        self._load_environment_variables(load_env)

    def _load_environment_variables(self, load_env: bool = True):
        """Load environment variables from .env.dist first, then .env if it exists.

        Args:
            load_env: Whether to load .env file. If False, only .env.dist is loaded.
        """
        # First load from .env.dist (default values)
        env_dist_path = Path(".env.dist")
        if env_dist_path.exists():
            load_dotenv(env_dist_path)

        if load_env:
            env_path = Path(".env")
            if env_path.exists():
                load_dotenv(env_path, override=True)

    def validate_required_env_vars(self):
        """Validate that all required environment variables are present based on provider."""
        provider = self.llm_provider

        if provider == "openai":
            if not self.openai_api_key:
                raise ConfigurationError(
                    "OPENAI_API_KEY not found in environment variables. " "Please set your OpenAI API key in .env file."
                )
        elif provider == "anthropic":
            if not self.anthropic_api_key:
                raise ConfigurationError(
                    "ANTHROPIC_API_KEY not found in environment variables. " "Please set your Anthropic API key in .env file."
                )
        elif provider == "openrouter":
            if not self.openrouter_api_key:
                raise ConfigurationError(
                    "OPENROUTER_API_KEY not found in environment variables. " "Please set your OpenRouter API key in .env file."
                )
        elif provider == "ollama":
            # Ollama doesn't require API key, but we should validate the base URL
            if not self.ollama_base_url:
                raise ConfigurationError(
                    "OLLAMA_BASE_URL not found in environment variables. " "Please set your Ollama base URL in .env file."
                )
        else:
            raise ConfigurationError(
                f"Unsupported LLM provider: {provider}. " f"Supported providers are: openai, anthropic, openrouter, ollama"
            )

    @property
    def llm_provider(self) -> LLMProvider:
        """Get the LLM provider from environment variables."""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if provider not in ["openai", "anthropic", "openrouter", "ollama"]:
            raise ConfigurationError(
                f"Invalid LLM_PROVIDER: {provider}. " f"Supported providers are: openai, anthropic, openrouter, ollama"
            )
        return provider  # type: ignore

    # OpenAI Configuration
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from environment variables."""
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def openai_model_name(self) -> str:
        """Get OpenAI model name from environment variables."""
        return os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-mini-2025-04-14")

    # Anthropic Configuration
    @property
    def anthropic_api_key(self) -> str:
        """Get Anthropic API key from environment variables."""
        return os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def anthropic_model_name(self) -> str:
        """Get Anthropic model name from environment variables."""
        return os.getenv("ANTHROPIC_MODEL_NAME", "claude-sonnet-4-20250514")

    # OpenRouter Configuration
    @property
    def openrouter_api_key(self) -> str:
        """Get OpenRouter API key from environment variables."""
        return os.getenv("OPENROUTER_API_KEY", "")

    @property
    def openrouter_model_name(self) -> str:
        """Get OpenRouter model name from environment variables."""
        return os.getenv("OPENROUTER_MODEL_NAME", "anthropic/claude-sonnet-4")

    @property
    def openrouter_base_url(self) -> str:
        """Get OpenRouter base URL from environment variables."""
        return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # Ollama Configuration
    @property
    def ollama_base_url(self) -> str:
        """Get Ollama base URL from environment variables."""
        return os.getenv("OLLAMA_BASE_URL", "")

    @property
    def ollama_model_name(self) -> str:
        """Get Ollama model name from environment variables."""
        return os.getenv("OLLAMA_MODEL_NAME", "qwen2.5vl:7b")

    @property
    def ollama_prompt_template(self) -> str:
        """Get Ollama-specific prompt template from environment variables."""
        default_ollama_prompt = """Look at the resonant‑angle plot (y=0–2π).
If all points stay inside one narrow band (<0.5 of the full height) or inside two opposite narrow horizontal bands with an empty gap between them, answer “resonant”.
If the points span most of the height—climbing in slanted stripes or scattered like pepper across the full y‑range—answer “non‑resonant”.
Ignore a few stray outliers and reply with exactly that single lowercase word.

Respond with JSON format:
{"status": "non-resonant", "subtype": "circulation"} for diagonal lines
{"status": "resonant", "subtype": "libration"} for oscillating patterns"""
        return os.getenv("OLLAMA_PROMPT_TEMPLATE", default_ollama_prompt)

    # Universal properties for backward compatibility
    @property
    def default_model_name(self) -> str:
        """Get the default model name based on the current provider."""
        provider = self.llm_provider
        if provider == "openai":
            return self.openai_model_name
        elif provider == "anthropic":
            return self.anthropic_model_name
        elif provider == "openrouter":
            return self.openrouter_model_name
        elif provider == "ollama":
            return self.ollama_model_name
        else:
            raise ConfigurationError(f"Unknown provider: {provider}")

    def get_prompt_template(self, env_var_name: str = "PROMPT_TEMPLATE") -> str:
        """Get a prompt template from the specified environment variable."""
        if not env_var_name or not env_var_name.strip():
            raise ConfigurationError("Prompt environment variable name cannot be empty.")

        value = os.getenv(env_var_name.strip())
        if not value:
            raise ConfigurationError(
                f"{env_var_name.strip()} is not set. Copy .env.dist to .env and configure the prompt template."
            )
        return value

    @property
    def prompt_template(self) -> str:
        """Get the default prompt template from environment variables."""
        return self.get_prompt_template("PROMPT_TEMPLATE")


# Global configuration instance
config = Config()
