"""JSON schema definitions for structured LLM outputs."""

from typing import Literal
from pydantic import BaseModel, Field


class LibrationAnalysisResult(BaseModel):
    """Structured result from libration analysis of resonant angle plots."""

    status: Literal["resonant", "non-resonant", "transient", "controversial"] = Field(
        description="Overall classification of the resonant angle behavior"
    )

    subtype: str = Field(
        description="Detailed subtype classification such as 'apocentric libration', 'apocentric circulation', 'double libration', 'circulation', or other descriptive text"
    )

    @classmethod
    def get_ollama_schema(cls) -> dict:
        """Get the schema formatted for Ollama structured outputs."""
        return cls.model_json_schema()


LIBRATION_ANALYSIS_SCHEMA = LibrationAnalysisResult
