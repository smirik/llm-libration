"""Type definitions for llm-libration package."""

from enum import Enum


class ResonanceType(Enum):
    """Enumeration of possible resonance analysis results."""

    RESONANT = "resonant"  # Pure libration (maps to 'pure' from LLM)
    NON_RESONANT = "non-resonant"  # Circulation most of the time
    CONTROVERSIAL = "controversial"  # Transient behavior (maps to 'transient' from LLM)

    def __str__(self) -> str:
        return self.value
