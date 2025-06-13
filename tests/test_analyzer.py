"""Tests for the ResonanceAnalyzer class."""

import os
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
from PIL import Image

from llm_libration.analyzer import ResonanceAnalyzer
from llm_libration.types import ResonanceType
from llm_libration.exceptions import ImageAnalysisError, LLMResponseError, ConfigurationError


class TestResonanceAnalyzer:
    """Test cases for ResonanceAnalyzer class."""

    @pytest.fixture
    def mock_openai_key(self):
        """Mock OpenAI API key environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            yield

    @pytest.fixture
    def sample_image(self):
        """Create a temporary sample image for testing."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create a simple test image
            img = Image.new('RGB', (100, 100), color='white')
            img.save(f.name, 'PNG')
            yield f.name
        # Cleanup
        os.unlink(f.name)

    @pytest.fixture
    def analyzer(self, mock_openai_key):
        """Create a ResonanceAnalyzer instance for testing."""
        with patch('llm_libration.analyzer.ChatOpenAI'):
            return ResonanceAnalyzer()

    def test_init_with_api_key(self, mock_openai_key):
        """Test successful initialization with API key."""
        with patch('llm_libration.analyzer.ChatOpenAI') as mock_chat:
            analyzer = ResonanceAnalyzer()
            assert analyzer is not None
            mock_chat.assert_called_once()

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="OPENAI_API_KEY not found"):
                ResonanceAnalyzer()

    def test_init_custom_parameters(self, mock_openai_key):
        """Test initialization with custom parameters."""
        with patch('llm_libration.analyzer.ChatOpenAI') as mock_chat:
            ResonanceAnalyzer(model_name="gpt-4", temperature=0.5)
            mock_chat.assert_called_once_with(model="gpt-4", temperature=0.5, max_tokens=50)

    def test_encode_image_success(self, analyzer, sample_image):
        """Test successful image encoding."""
        result = analyzer._encode_image(sample_image)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_encode_image_nonexistent_file(self, analyzer):
        """Test encoding of non-existent image file."""
        with pytest.raises(ImageAnalysisError, match="Image file not found"):
            analyzer._encode_image("nonexistent.png")

    def test_encode_image_invalid_file(self, analyzer):
        """Test encoding of invalid image file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not an image")
            f.flush()

            with pytest.raises(ImageAnalysisError, match="Failed to encode image"):
                analyzer._encode_image(f.name)

            os.unlink(f.name)

    @pytest.mark.parametrize(
        "llm_response,expected_type",
        [
            ("pure", ResonanceType.RESONANT),
            ("PURE", ResonanceType.RESONANT),
            ("transient", ResonanceType.CONTROVERSIAL),
            ("TRANSIENT", ResonanceType.CONTROVERSIAL),
            ("non-resonant", ResonanceType.NON_RESONANT),
            ("NON-RESONANT", ResonanceType.NON_RESONANT),
            ("i do not know", ResonanceType.CONTROVERSIAL),
            ("I DO NOT KNOW", ResonanceType.CONTROVERSIAL),
        ],
    )
    def test_parse_llm_response_valid(self, analyzer, llm_response, expected_type):
        """Test parsing of valid LLM responses."""
        result = analyzer._parse_llm_response(llm_response)
        assert result == expected_type

    def test_parse_llm_response_invalid(self, analyzer):
        """Test parsing of invalid LLM response."""
        with pytest.raises(LLMResponseError, match="Unexpected LLM response"):
            analyzer._parse_llm_response("invalid response")

    def test_analyze_image_success(self, analyzer, sample_image):
        """Test successful image analysis."""
        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = "pure"
        analyzer.llm.invoke = Mock(return_value=mock_response)

        result = analyzer.analyze_image(sample_image)

        assert result == ResonanceType.RESONANT
        analyzer.llm.invoke.assert_called_once()

    def test_analyze_image_llm_error(self, analyzer, sample_image):
        """Test image analysis with LLM error."""
        # Mock the LLM to raise an exception
        analyzer.llm.invoke = Mock(side_effect=Exception("LLM error"))

        with pytest.raises(ImageAnalysisError, match="Unexpected error during image analysis"):
            analyzer.analyze_image(sample_image)

    def test_analyze_image_invalid_response(self, analyzer, sample_image):
        """Test image analysis with invalid LLM response."""
        # Mock the LLM response with invalid content
        mock_response = Mock()
        mock_response.content = "invalid"
        analyzer.llm.invoke = Mock(return_value=mock_response)

        with pytest.raises(LLMResponseError):
            analyzer.analyze_image(sample_image)

    def test_analyze_image_pathlib_path(self, analyzer, sample_image):
        """Test image analysis with pathlib.Path input."""
        mock_response = Mock()
        mock_response.content = "transient"
        analyzer.llm.invoke = Mock(return_value=mock_response)

        result = analyzer.analyze_image(Path(sample_image))

        assert result == ResonanceType.CONTROVERSIAL

    def test_prompt_template_exists(self, analyzer):
        """Test that the prompt template is properly defined."""
        assert hasattr(analyzer, 'PROMPT_TEMPLATE')
        assert isinstance(analyzer.PROMPT_TEMPLATE, str)
        assert len(analyzer.PROMPT_TEMPLATE) > 100  # Should be a substantial prompt
        assert "astronomer" in analyzer.PROMPT_TEMPLATE.lower()
        assert "pure" in analyzer.PROMPT_TEMPLATE.lower()
        assert "transient" in analyzer.PROMPT_TEMPLATE.lower()
        assert "non-resonant" in analyzer.PROMPT_TEMPLATE.lower()


class TestResonanceType:
    """Test cases for ResonanceType enum."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        assert ResonanceType.RESONANT.value == "resonant"
        assert ResonanceType.NON_RESONANT.value == "non-resonant"
        assert ResonanceType.CONTROVERSIAL.value == "controversial"

    def test_string_representation(self):
        """Test string representation of enum values."""
        assert str(ResonanceType.RESONANT) == "resonant"
        assert str(ResonanceType.NON_RESONANT) == "non-resonant"
        assert str(ResonanceType.CONTROVERSIAL) == "controversial"


class TestIntegration:
    """Integration tests."""

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY environment variable")
    def test_real_api_integration(self):
        """Test with real OpenAI API (requires API key)."""
        # This test only runs if OPENAI_API_KEY is available
        # Create a simple test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new('RGB', (400, 300), color='white')
            img.save(f.name, 'PNG')

            try:
                analyzer = ResonanceAnalyzer()
                # This might fail due to the simple test image, but shouldn't crash
                result = analyzer.analyze_image(f.name)
                assert isinstance(result, ResonanceType)
            except (LLMResponseError, ImageAnalysisError):
                # These are acceptable for a simple test image
                pass
            finally:
                os.unlink(f.name)
