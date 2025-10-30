"""JSON schema definitions for structured LLM outputs."""

from pydantic import BaseModel, Field

from ..types import ResonanceType


class LibrationAnalysisResult(BaseModel):
    """Structured result from libration analysis of resonant angle plots."""

    status: ResonanceType = Field(description="Classification of the resonant angle behavior")

    @classmethod
    def get_ollama_schema(cls) -> dict:
        """Get the schema formatted for Ollama structured outputs."""
        return cls.model_json_schema()


LIBRATION_ANALYSIS_SCHEMA = LibrationAnalysisResult
