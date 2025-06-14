"""Tests for type definitions and enums."""

from llm_libration.types import ResonanceType


class TestResonanceType:
    """Test cases for ResonanceType enum."""

    def test_enum_values(self):
        """Test that ResonanceType has expected values."""
        assert ResonanceType.RESONANT.value == "resonant"
        assert ResonanceType.NON_RESONANT.value == "non-resonant"
        assert ResonanceType.CONTROVERSIAL.value == "controversial"

    def test_string_representation(self):
        """Test string representation of ResonanceType."""
        assert str(ResonanceType.RESONANT) == "resonant"
        assert str(ResonanceType.NON_RESONANT) == "non-resonant"
        assert str(ResonanceType.CONTROVERSIAL) == "controversial"

    def test_enum_membership(self):
        """Test enum membership and iteration."""
        all_types = list(ResonanceType)
        assert len(all_types) == 3
        assert ResonanceType.RESONANT in all_types
        assert ResonanceType.NON_RESONANT in all_types
        assert ResonanceType.CONTROVERSIAL in all_types

    def test_enum_equality(self):
        """Test enum equality comparisons."""
        assert ResonanceType.RESONANT == ResonanceType.RESONANT
        assert ResonanceType.RESONANT != ResonanceType.NON_RESONANT
        assert ResonanceType.RESONANT != ResonanceType.CONTROVERSIAL
