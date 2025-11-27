"""Main analyzer module for resonant angle libration detection."""

import logging
from pathlib import Path
from typing import Optional, Union

from .types import ResonanceType
from .exceptions import ImageAnalysisError, LLMResponseError, ConfigurationError
from .config import config
from .llm import LLMClient
from .llm.schema import LibrationAnalysisResult

logger = logging.getLogger(__name__)


class LibrationAnalyzer:
    """Analyzer for detecting libration patterns in resonant angle plots using LLMs."""

    def __init__(self, model_name: str = None, provider: str = None, quantization: str = None):
        """
        Initialize the LibrationAnalyzer.

        Args:
            model_name: Name of the model to use (defaults to config value for current provider)
            provider: LLM provider to use (defaults to config value)
            quantization: For HuggingFace only: none, 4bit, or 8bit (defaults to config value)
        """
        # Resolve provider first, then validate for that specific provider
        provider = provider or config.llm_provider
        config.validate_required_env_vars(provider)

        api_key = ""
        base_url = ""
        quant = "none"
        device_map = "auto"
        torch_dtype = "bfloat16"

        if provider == "openai":
            api_key, model_name = config.openai_api_key, model_name or config.openai_model_name
        elif provider == "anthropic":
            api_key, model_name = config.anthropic_api_key, model_name or config.anthropic_model_name
        elif provider == "openrouter":
            api_key, model_name, base_url = (
                config.openrouter_api_key,
                model_name or config.openrouter_model_name,
                config.openrouter_base_url,
            )
        elif provider == "ollama":
            model_name, base_url = model_name or config.ollama_model_name, config.ollama_base_url
        elif provider == "huggingface":
            model_name = model_name or config.hf_model_name
            quant = quantization or config.hf_quantization
            device_map = config.hf_device_map
            torch_dtype = config.hf_torch_dtype
        elif provider == "mlx":
            model_name = model_name or config.mlx_model_name
        else:
            raise ConfigurationError(f"Unsupported provider: {provider}")

        self.llm_client = LLMClient(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            quantization=quant,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

    MAX_RETRIES = 3

    def analyze_image(self, image_path: Union[str, Path], prompt: Optional[str] = None) -> LibrationAnalysisResult:
        """
        Analyze a resonant angle plot image and return detailed structured result.

        Args:
            image_path: Path to the image file containing the resonant angle plot

        Returns:
            LibrationAnalysisResult with status and subtype information

        Raises:
            ImageAnalysisError: If image cannot be processed
            LLMResponseError: If LLM response is invalid
            ConfigurationError: If configuration is missing
        """
        prompt_template = prompt or config.prompt_template
        last_error: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return self.llm_client.analyze_image_with_prompt(image_path, prompt_template)
            except LLMResponseError as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES and self._is_retryable_error(exc):
                    logger.warning(
                        f"Retryable error on attempt {attempt}/{self.MAX_RETRIES}: {exc}. Retrying..."
                    )
                    continue
                raise
            except (ImageAnalysisError, ConfigurationError):
                raise
            except Exception as exc:  # pragma: no cover - unexpected edge cases
                raise ImageAnalysisError(f"Unexpected error during image analysis: {str(exc)}")

        # Should not reach here, but safety
        if last_error:
            raise last_error
        raise ImageAnalysisError("Unable to analyze image due to unknown error.")

    def get_resonance_type(self, image_path: Union[str, Path], prompt: Optional[str] = None) -> ResonanceType:
        """
        Analyze a resonant angle plot image and return the ResonanceType enum.

        This is a convenience method that extracts just the status from the structured result.

        Args:
            image_path: Path to the image file containing the resonant angle plot

        Returns:
            ResonanceType enum indicating the type of resonance behavior

        Raises:
            ImageAnalysisError: If image cannot be processed
            LLMResponseError: If LLM response is invalid
            ConfigurationError: If configuration is missing
        """
        try:
            result = self.analyze_image(image_path, prompt=prompt)
            return result.status
        except (ImageAnalysisError, LLMResponseError, ConfigurationError):
            raise
        except Exception as e:
            raise ImageAnalysisError(f"Unexpected error during image analysis: {str(e)}")

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        """Detect transient LLM failures that are worth retrying."""
        message = str(exc).lower()
        transient_keywords = [
            "length limit was reached",
            "completionusage",
            "connection error",
            "timed out",
            "temporarily unavailable",
            "rate limit",
        ]
        return any(keyword in message for keyword in transient_keywords)
