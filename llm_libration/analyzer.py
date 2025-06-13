"""Main analyzer module for resonant angle libration detection."""

from pathlib import Path
from typing import Union

from .types import ResonanceType
from .exceptions import ImageAnalysisError, LLMResponseError, ConfigurationError
from .config import config
from .llm import LLMClient


class LibrationAnalyzer:
    """Analyzer for detecting libration patterns in resonant angle plots using LLMs."""

    def __init__(self, model_name: str = None, provider: str = None):
        """
        Initialize the LibrationAnalyzer.

        Args:
            model_name: Name of the model to use (defaults to config value for current provider)
            provider: LLM provider to use (defaults to config value)
        """
        # Validate configuration
        config.validate_required_env_vars()

        # Use config defaults if not provided
        provider = provider or config.llm_provider

        # Get provider-specific configuration
        api_key = ""
        base_url = ""
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
        else:
            raise ConfigurationError(f"Unsupported provider: {provider}")

        # Initialize the LLM client
        self.llm_client = LLMClient(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )

    def _parse_llm_response(self, response: str) -> ResonanceType:
        """
        Parse the LLM response and map it to ResonanceType.

        Args:
            response: Raw response from the LLM

        Returns:
            ResonanceType enum value

        Raises:
            LLMResponseError: If response cannot be parsed
        """
        response = response.strip().lower()

        # Check for exact matches first
        if response == "pure":
            return ResonanceType.RESONANT
        elif response == "transient":
            return ResonanceType.CONTROVERSIAL
        elif response == "non-resonant":
            return ResonanceType.NON_RESONANT
        elif response == "i do not know":
            return ResonanceType.CONTROVERSIAL  # Treat uncertainty as controversial

        # Handle cases where the response contains the keyword but with additional text
        # (common with Ollama or when additional notes are included)
        elif response.startswith("pure"):
            return ResonanceType.RESONANT
        elif response.startswith("transient"):
            return ResonanceType.CONTROVERSIAL
        elif response.startswith("non-resonant"):
            return ResonanceType.NON_RESONANT
        elif response.startswith("i do not know"):
            return ResonanceType.CONTROVERSIAL
        else:
            raise LLMResponseError(f"Unexpected LLM response: {response}")

    def analyze_image(self, image_path: Union[str, Path]) -> ResonanceType:
        """
        Analyze a resonant angle plot image to determine libration type.

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
            raw_response = self.llm_client.analyze_image_with_prompt(image_path, config.prompt_template)
            return self._parse_llm_response(raw_response)
        except (ImageAnalysisError, LLMResponseError, ConfigurationError):
            raise
        except Exception as e:
            raise ImageAnalysisError(f"Unexpected error during image analysis: {str(e)}")
