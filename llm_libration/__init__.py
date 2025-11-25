"""
LLM Libration - Identify librations of resonant angles by LLMs.

This package provides functionality to analyze astronomical resonant angle plots
using Large Language Models to determine if the angles show libration patterns.
"""

__version__ = "0.1.0"
__author__ = "Evgeny Smirnov"

from .analyzer import LibrationAnalyzer
from .types import ResonanceType, simplify_resonance_type
from .exceptions import LLMLibrationError, ImageAnalysisError
from .config import config
from .llm import LLMClient

__all__ = [
    "LibrationAnalyzer",
    "ResonanceType",
    "simplify_resonance_type",
    "LLMLibrationError",
    "ImageAnalysisError",
    "config",
    "LLMClient",
]
