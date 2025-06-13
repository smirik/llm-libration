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
        return os.getenv("OLLAMA_MODEL_NAME", "gemma3")

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

    @property
    def prompt_template(self) -> str:
        """Get the prompt template from environment variables or use default."""
        default_prompt = (
            """I want you to act a scientist–astronomer. You will get an image uploaded. """
            """The image contains the plot of the resonant angle of an asteroid vs time (from 0 to 100000 years). """
            """The limits of OY axis are -pi and pi. The resonant angle cannot exceed these limits.

It is known that if the resonant angle librates, then the asteroid is trapped in the resonance. """
            """Librations mean oscillations, like sine. It means that the curve is within some limits (i.e., +2, or +1) """
            """and does not come close to the borders (-pi and pi).

The opposite situation is when the resonant angle circulates. """
            """It means that the curve is not limited and can reach the borders of the plot. """
            """In our case, if the resonant angle is greater than pi or less than -pi, then we add or substract 2pi to the """
            """resonant angle to make it within the limits. Therefore, in the case of circulation, the pattern will be """
            """like linear curves parallel each other.

I want you to assess visually whether the resonant angle librates if you were a human looking at this image.

There are three possible cases:

1. The resonant angle librates all the time (from 0 to 100000). Then you should reply 'pure'.
2. The resonant angle could librate some significant time, but in other time is circulates. """
            """Let's assume that by significant I mean 20000 years. In this case, you should write 'transient'.
3. Otherwise, when the resonant angle circulates most of the time, please write 'non-resonant'.

As output, I want you only to print one word: pure, transient, or non-resonant. """
            """If you are not sure, write 'I do not know'. You will get tips if you perform the identification correctly."""
        )
        return os.getenv("PROMPT_TEMPLATE", default_prompt)


# Global configuration instance
config = Config()
