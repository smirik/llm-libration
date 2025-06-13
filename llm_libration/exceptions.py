"""Custom exceptions for llm-libration package."""


class LLMLibrationError(Exception):
    """Base exception for llm-libration package."""

    pass


class ImageAnalysisError(LLMLibrationError):
    """Exception raised when image analysis fails."""

    pass


class LLMResponseError(LLMLibrationError):
    """Exception raised when LLM response is invalid or unexpected."""

    pass


class ConfigurationError(LLMLibrationError):
    """Exception raised when configuration is missing or invalid."""

    pass
