"""
LLM Libration - Identify librations of resonant angles by LLMs.

This package provides functionality to analyze astronomical resonant angle plots
using Large Language Models to determine if the angles show libration patterns.
"""

__version__ = "0.1.0"
__author__ = "Evgeny Smirnov"

from .analyzer import ResonanceAnalyzer
from .types import ResonanceType
from .exceptions import LLMLibrationError, ImageAnalysisError

__all__ = [
    "ResonanceAnalyzer",
    "ResonanceType",
    "LLMLibrationError",
    "ImageAnalysisError",
]
