"""Tests for schema definitions and data models."""

import pytest
from llm_libration.llm.schema import (
    LibrationAnalysisResult,
    ResonantSubtype,
    NonResonantSubtype,
    TransientSubtype,
)


class TestSubtypeEnums:
    """Test cases for new subtype enums."""

    def test_resonant_subtype_enum_values(self):
        """Test that ResonantSubtype has expected values."""
        assert ResonantSubtype.CLEAR_LIBRATION.value == "clear libration"
        assert ResonantSubtype.APOCENTRIC_LIBRATION.value == "apocentric libration"
        assert ResonantSubtype.HIGH_AMPLITUDE_LIBRATION.value == "high amplitude libration"
        assert ResonantSubtype.LONG_PERIOD_LIBRATION.value == "long period libration"
        assert ResonantSubtype.NOISY_LIBRATION.value == "noisy libration"
        assert ResonantSubtype.DOUBLE_LIBRATION.value == "double libration"

    def test_non_resonant_subtype_enum_values(self):
        """Test that NonResonantSubtype has expected values."""
        assert NonResonantSubtype.CIRCULATION.value == "circulation"
        assert NonResonantSubtype.SPARSE_CIRCULATION.value == "sparse circulation"
        assert NonResonantSubtype.CHAOTIC_CIRCULATION.value == "chaotic circulation"
        assert NonResonantSubtype.MIXED_CIRCULATION.value == "mixed circulation"

    def test_transient_subtype_enum_values(self):
        """Test that TransientSubtype has expected values."""
        assert TransientSubtype.NOISY_LIBRATION_TO_CIRCULATION.value == "noisy libration to circulation"
        assert TransientSubtype.APOCENTRIC_WITH_CIRCULATION.value == "apocentric libration with circulation phases"
        assert TransientSubtype.ALTERNATING.value == "alternating libration and circulation"

    def test_enum_string_inheritance(self):
        """Test that enums inherit from str for JSON compatibility."""
        # All enums should inherit from str
        assert isinstance(ResonantSubtype.CLEAR_LIBRATION, str)
        assert isinstance(NonResonantSubtype.CIRCULATION, str)
        assert isinstance(TransientSubtype.ALTERNATING, str)


class TestLibrationAnalysisResult:
    """Test cases for LibrationAnalysisResult model."""

    def test_libration_analysis_result_with_enum_subtypes(self):
        """Test LibrationAnalysisResult works with enum subtypes."""
        # Test with ResonantSubtype
        result1 = LibrationAnalysisResult(status="resonant", subtype=ResonantSubtype.APOCENTRIC_LIBRATION)
        assert result1.status == "resonant"
        assert result1.subtype == ResonantSubtype.APOCENTRIC_LIBRATION
        assert result1.subtype == "apocentric libration"  # Should also work with string comparison

        # Test with NonResonantSubtype
        result2 = LibrationAnalysisResult(status="non-resonant", subtype=NonResonantSubtype.CIRCULATION)
        assert result2.status == "non-resonant"
        assert result2.subtype == NonResonantSubtype.CIRCULATION
        assert result2.subtype == "circulation"

        # Test with TransientSubtype
        result3 = LibrationAnalysisResult(status="transient", subtype=TransientSubtype.ALTERNATING)
        assert result3.status == "transient"
        assert result3.subtype == TransientSubtype.ALTERNATING
        assert result3.subtype == "alternating libration and circulation"

        # Test with string (for controversial cases)
        result4 = LibrationAnalysisResult(status="controversial", subtype="unclear pattern with mixed behavior")
        assert result4.status == "controversial"
        assert result4.subtype == "unclear pattern with mixed behavior"

    def test_libration_analysis_result_with_string_subtypes(self):
        """Test LibrationAnalysisResult works with string subtypes that match enum values."""
        # Test creating with string that matches enum value
        result = LibrationAnalysisResult(status="resonant", subtype="apocentric libration")
        assert result.status == "resonant"
        assert result.subtype == "apocentric libration"
        # It should still be a string, not automatically converted to enum
        assert isinstance(result.subtype, str)

    def test_libration_analysis_result_status_validation(self):
        """Test that LibrationAnalysisResult validates status field."""
        # Valid statuses should work
        valid_statuses = ["resonant", "non-resonant", "transient", "controversial"]
        for status in valid_statuses:
            result = LibrationAnalysisResult(status=status, subtype="test subtype")
            assert result.status == status

        # Invalid status should raise validation error
        with pytest.raises(ValueError):
            LibrationAnalysisResult(status="invalid_status", subtype="test subtype")

    def test_libration_analysis_result_json_serialization(self):
        """Test JSON serialization and deserialization."""
        # Test with enum subtype
        result_enum = LibrationAnalysisResult(status="resonant", subtype=ResonantSubtype.APOCENTRIC_LIBRATION)
        json_data = result_enum.model_dump()
        assert json_data["status"] == "resonant"
        assert json_data["subtype"] == ResonantSubtype.APOCENTRIC_LIBRATION

        # Test with string subtype
        result_str = LibrationAnalysisResult(status="controversial", subtype="unclear pattern")
        json_data_str = result_str.model_dump()
        assert json_data_str["status"] == "controversial"
        assert json_data_str["subtype"] == "unclear pattern"

    def test_get_ollama_schema(self):
        """Test that get_ollama_schema method returns valid schema."""
        schema = LibrationAnalysisResult.get_ollama_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "status" in schema["properties"]
        assert "subtype" in schema["properties"]
