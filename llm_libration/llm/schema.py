"""JSON schema definitions for structured LLM outputs."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from ..types import ResonanceType


class ResonantSubtype(str, Enum):
    CLEAR_LIBRATION = "clear libration"
    APOCENTRIC_LIBRATION = "apocentric libration"
    HIGH_AMPLITUDE_LIBRATION = "high amplitude libration"
    LONG_PERIOD_LIBRATION = "long period libration"
    NOISY_LIBRATION = "noisy libration"
    DOUBLE_LIBRATION = "double libration"


class NonResonantSubtype(str, Enum):
    CIRCULATION = "circulation"
    SPARSE_CIRCULATION = "sparse circulation"
    CHAOTIC_CIRCULATION = "chaotic circulation"
    MIXED_CIRCULATION = "mixed circulation"


class TransientSubtype(str, Enum):
    NOISY_LIBRATION_TO_CIRCULATION = "noisy libration to circulation"
    APOCENTRIC_WITH_CIRCULATION = "apocentric libration with circulation phases"
    ALTERNATING = "alternating libration and circulation"


class LibrationAnalysisResult(BaseModel):
    """Structured result from libration analysis of resonant angle plots."""

    status: ResonanceType = Field(description="Classification of the resonant angle behavior")
    subtype: Optional[str] = Field(
        default=None,
        description="Optional subtype description (enum or string) for the detected pattern",
    )

    @classmethod
    def get_ollama_schema(cls) -> dict:
        """Get the schema formatted for Ollama structured outputs."""
        return cls.model_json_schema()


LIBRATION_ANALYSIS_SCHEMA = LibrationAnalysisResult
