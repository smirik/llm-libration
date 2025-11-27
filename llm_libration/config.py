"""Configuration management for the LLM Libration project."""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from .exceptions import ConfigurationError

# Supported LLM providers
LLMProvider = Literal["openai", "anthropic", "openrouter", "ollama", "huggingface", "mlx"]


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

    def validate_required_env_vars(self, provider: str | None = None):
        """Validate that all required environment variables are present based on provider.

        Args:
            provider: Provider to validate. If None, uses the default from LLM_PROVIDER env var.
        """
        provider = provider or self.llm_provider

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
            if not self.ollama_base_url:
                raise ConfigurationError(
                    "OLLAMA_BASE_URL not found in environment variables. " "Please set your Ollama base URL in .env file."
                )
        elif provider == "huggingface":
            pass
        elif provider == "mlx":
            import platform
            if platform.system() != "Darwin":
                raise ConfigurationError(
                    "MLX provider is only available on macOS. "
                    "Use 'huggingface' or 'ollama' provider on other platforms."
                )
            if platform.machine() not in ("arm64", "aarch64"):
                raise ConfigurationError(
                    "MLX provider requires Apple Silicon (M1/M2/M3/M4). "
                    "Use 'huggingface' or 'ollama' provider on Intel Macs."
                )
        else:
            raise ConfigurationError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers are: openai, anthropic, openrouter, ollama, huggingface, mlx"
            )

    @property
    def llm_provider(self) -> LLMProvider:
        """Get the LLM provider from environment variables."""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if provider not in ["openai", "anthropic", "openrouter", "ollama", "huggingface", "mlx"]:
            raise ConfigurationError(
                f"Invalid LLM_PROVIDER: {provider}. "
                f"Supported providers are: openai, anthropic, openrouter, ollama, huggingface, mlx"
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
If all points stay inside one narrow band (<0.5 of the full height) or inside two opposite narrow horizontal bands with an empty gap between them, answer "resonant".
If the points span most of the height—climbing in slanted stripes or scattered like pepper across the full y‑range—answer "non‑resonant".
Ignore a few stray outliers and reply with exactly that single lowercase word.

Respond with JSON format:
{"status": "non-resonant", "subtype": "circulation"} for diagonal lines
{"status": "resonant", "subtype": "libration"} for oscillating patterns"""
        return os.getenv("OLLAMA_PROMPT_TEMPLATE", default_ollama_prompt)

    # HuggingFace Configuration
    @property
    def hf_model_name(self) -> str:
        """Get HuggingFace model name."""
        return os.getenv("HF_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")

    @property
    def hf_quantization(self) -> str:
        """Get quantization mode: none, 4bit, or 8bit."""
        return os.getenv("HF_QUANTIZATION", "none").lower()

    @property
    def hf_device_map(self) -> str:
        """Get device mapping strategy."""
        return os.getenv("HF_DEVICE_MAP", "auto")

    @property
    def hf_torch_dtype(self) -> str:
        """Get torch dtype for model loading."""
        return os.getenv("HF_TORCH_DTYPE", "bfloat16")

    @property
    def hf_max_new_tokens(self) -> int:
        """Get max new tokens for generation."""
        return int(os.getenv("HF_MAX_NEW_TOKENS", "256"))

    @property
    def hf_4bit_quant_type(self) -> str:
        """Get 4-bit quantization type: nf4 or fp4."""
        return os.getenv("HF_4BIT_QUANT_TYPE", "nf4")

    @property
    def hf_4bit_use_double_quant(self) -> bool:
        """Whether to use double quantization for 4-bit."""
        return os.getenv("HF_4BIT_USE_DOUBLE_QUANT", "true").lower() == "true"

    @property
    def hf_4bit_compute_dtype(self) -> str:
        """Get compute dtype for 4-bit quantization."""
        return os.getenv("HF_4BIT_COMPUTE_DTYPE", "bfloat16")

    # MLX Configuration (macOS Apple Silicon only)
    @property
    def mlx_model_name(self) -> str:
        """Get MLX model name (from mlx-community)."""
        return os.getenv("MLX_MODEL_NAME", "mlx-community/gemma-3n-E2B-it-4bit")

    @property
    def mlx_max_tokens(self) -> int:
        """Get max new tokens for MLX generation."""
        return int(os.getenv("MLX_MAX_TOKENS", "256"))

    @property
    def mlx_temperature(self) -> float:
        """Get temperature for MLX generation."""
        return float(os.getenv("MLX_TEMPERATURE", "0.0"))

    @property
    def mlx_verbose(self) -> bool:
        """Whether to enable verbose MLX output."""
        return os.getenv("MLX_VERBOSE", "false").lower() == "true"

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
        elif provider == "huggingface":
            return self.hf_model_name
        elif provider == "mlx":
            return self.mlx_model_name
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
