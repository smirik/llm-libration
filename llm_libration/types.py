"""Type definitions for llm-libration package."""

from enum import Enum


class ResonanceType(Enum):
    """Enumeration of possible resonance analysis results."""

    RESONANT = "resonant"  # Pure libration (maps to 'resonant' from LLM)
    NON_RESONANT = "non-resonant"  # Circulation most of the time
    TRANSIENT = "transient"  # Transient behavior between resonant and non-resonant
    CONTROVERSIAL = "controversial"  # Unclear or ambiguous behavior

    def __str__(self) -> str:
        return self.value


def simplify_resonance_type(value: ResonanceType) -> ResonanceType:
    """Collapse detailed resonance states into binary resonant/non-resonant classes."""
    if value in (ResonanceType.RESONANT, ResonanceType.TRANSIENT):
        return ResonanceType.RESONANT
    return ResonanceType.NON_RESONANT
