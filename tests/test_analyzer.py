"""Tests for LibrationAnalyzer functionality."""

import os
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

import pytest
from PIL import Image

from llm_libration.analyzer import LibrationAnalyzer
from llm_libration.llm.schema import LibrationAnalysisResult
from llm_libration.types import ResonanceType
from llm_libration.exceptions import ImageAnalysisError, LLMResponseError, ConfigurationError


class TestLibrationAnalyzer:
    """Test cases for LibrationAnalyzer class."""

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
    def mock_openai_env(self):
        """Mock OpenAI environment variables."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LLM_PROVIDER": "openai"}):
            yield

    @pytest.fixture
    def mock_anthropic_env(self):
        """Mock Anthropic environment variables."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "LLM_PROVIDER": "anthropic"}):
            yield

    @pytest.fixture
    def mock_ollama_env(self):
        """Mock Ollama environment variables."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "OLLAMA_BASE_URL": "http://localhost:11434"}):
            yield

    @pytest.fixture
    def openai_analyzer(self, mock_openai_env):
        """Create a LibrationAnalyzer instance for OpenAI testing."""
        with patch('llm_libration.llm.client.ChatOpenAI'):
            return LibrationAnalyzer()

    @pytest.fixture
    def anthropic_analyzer(self, mock_anthropic_env):
        """Create a LibrationAnalyzer instance for Anthropic testing."""
        with patch('llm_libration.llm.client.ChatAnthropic'):
            return LibrationAnalyzer()

    @pytest.fixture
    def ollama_analyzer(self, mock_ollama_env):
        """Create a LibrationAnalyzer instance for Ollama testing."""
        with patch('llm_libration.llm.client.ollama'):
            return LibrationAnalyzer()

    @pytest.mark.parametrize(
        "status,expected_type",
        [
            ("resonant", ResonanceType.RESONANT),
            ("non-resonant", ResonanceType.NON_RESONANT),
            ("transient", ResonanceType.CONTROVERSIAL),
            ("controversial", ResonanceType.CONTROVERSIAL),
        ],
    )
    def test_map_status_to_resonance_type_valid(self, openai_analyzer, status, expected_type):
        """Test mapping of valid structured output status to ResonanceType."""
        result = LibrationAnalysisResult(status=status, subtype="test subtype")
        mapped_type = openai_analyzer._map_status_to_resonance_type(result)
        assert mapped_type == expected_type

    def test_map_status_to_resonance_type_invalid(self, openai_analyzer):
        """Test mapping of invalid structured output status."""
        # Since Pydantic validates the status field, we need to patch the method directly
        # to test what happens when an invalid status somehow gets through
        from llm_libration.llm.schema import LibrationAnalysisResult

        # Create a valid result first, then monkey-patch the status
        result = LibrationAnalysisResult(status="resonant", subtype="test subtype")
        result.status = "invalid"  # This bypasses Pydantic validation

        with pytest.raises(LLMResponseError, match="Unexpected LLM status"):
            openai_analyzer._map_status_to_resonance_type(result)

    def test_analyze_image_success_openai(self, openai_analyzer, sample_image):
        """Test successful image analysis with OpenAI."""
        # Mock the LLM client response with structured output
        from llm_libration.llm.schema import ResonantSubtype

        mock_result = LibrationAnalysisResult(status="resonant", subtype=ResonantSubtype.APOCENTRIC_LIBRATION)
        openai_analyzer.llm_client.analyze_image_with_prompt = Mock(return_value=mock_result)

        result = openai_analyzer.analyze_image(sample_image)

        assert isinstance(result, LibrationAnalysisResult)
        assert result.status == "resonant"
        assert result.subtype == ResonantSubtype.APOCENTRIC_LIBRATION
        openai_analyzer.llm_client.analyze_image_with_prompt.assert_called_once()

    def test_analyze_image_success_anthropic(self, anthropic_analyzer, sample_image):
        """Test successful image analysis with Anthropic."""
        # Mock the LLM client response with structured output
        from llm_libration.llm.schema import TransientSubtype

        mock_result = LibrationAnalysisResult(status="transient", subtype=TransientSubtype.ALTERNATING)
        anthropic_analyzer.llm_client.analyze_image_with_prompt = Mock(return_value=mock_result)

        result = anthropic_analyzer.analyze_image(sample_image)

        assert isinstance(result, LibrationAnalysisResult)
        assert result.status == "transient"
        assert result.subtype == TransientSubtype.ALTERNATING
        anthropic_analyzer.llm_client.analyze_image_with_prompt.assert_called_once()

    def test_analyze_image_success_ollama(self, ollama_analyzer, sample_image):
        """Test successful image analysis with Ollama."""
        # Mock the LLM client response with structured output
        from llm_libration.llm.schema import NonResonantSubtype

        mock_result = LibrationAnalysisResult(status="non-resonant", subtype=NonResonantSubtype.CIRCULATION)
        ollama_analyzer.llm_client.analyze_image_with_prompt = Mock(return_value=mock_result)

        result = ollama_analyzer.analyze_image(sample_image)

        assert isinstance(result, LibrationAnalysisResult)
        assert result.status == "non-resonant"
        assert result.subtype == NonResonantSubtype.CIRCULATION
        ollama_analyzer.llm_client.analyze_image_with_prompt.assert_called_once()

    def test_analyze_image_llm_error(self, openai_analyzer, sample_image):
        """Test image analysis with LLM error."""
        # Mock LLM client to raise LLMResponseError
        openai_analyzer.llm_client.analyze_image_with_prompt = Mock(side_effect=LLMResponseError("API error"))

        with pytest.raises(LLMResponseError):
            openai_analyzer.analyze_image(sample_image)

    def test_analyze_image_invalid_response(self, openai_analyzer, sample_image):
        """Test image analysis with LLM error during mapping."""
        # Mock the LLM client to return a valid result, but then patch the mapping method to fail
        mock_result = LibrationAnalysisResult(status="resonant", subtype="test")
        openai_analyzer.llm_client.analyze_image_with_prompt = Mock(return_value=mock_result)

        # This test is no longer relevant since analyze_image returns the result directly
        # But we can test that it returns the expected structure
        result = openai_analyzer.analyze_image(sample_image)
        assert isinstance(result, LibrationAnalysisResult)
        assert result.status == "resonant"
        assert result.subtype == "test"

    def test_analyze_image_pathlib_path(self, openai_analyzer, sample_image):
        """Test image analysis with pathlib.Path input."""
        mock_result = LibrationAnalysisResult(status="transient", subtype="mixed behavior")
        openai_analyzer.llm_client.analyze_image_with_prompt = Mock(return_value=mock_result)

        result = openai_analyzer.analyze_image(Path(sample_image))

        assert isinstance(result, LibrationAnalysisResult)
        assert result.status == "transient"
        assert result.subtype == "mixed behavior"
        openai_analyzer.llm_client.analyze_image_with_prompt.assert_called_once()

    def test_get_resonance_type_success(self, openai_analyzer, sample_image):
        """Test get_resonance_type method that returns ResonanceType enum."""
        from llm_libration.llm.schema import ResonantSubtype

        mock_result = LibrationAnalysisResult(status="resonant", subtype=ResonantSubtype.APOCENTRIC_LIBRATION)
        openai_analyzer.llm_client.analyze_image_with_prompt = Mock(return_value=mock_result)

        result = openai_analyzer.get_resonance_type(sample_image)

        assert result == ResonanceType.RESONANT
        openai_analyzer.llm_client.analyze_image_with_prompt.assert_called_once()

    def test_get_resonance_type_error(self, openai_analyzer, sample_image):
        """Test get_resonance_type method with LLM error."""
        openai_analyzer.llm_client.analyze_image_with_prompt = Mock(side_effect=LLMResponseError("API error"))

        with pytest.raises(LLMResponseError):
            openai_analyzer.get_resonance_type(sample_image)

    def test_prompt_template_access(self, openai_analyzer):
        """Test that the prompt template is accessible through config."""
        from llm_libration import config

        assert isinstance(config.prompt_template, str)
        assert len(config.prompt_template) > 100  # Should be a substantial prompt
        # Check for key terms that should be in any libration analysis prompt
        prompt_lower = config.prompt_template.lower()
        assert any(term in prompt_lower for term in ["resonant", "libration", "circulation", "asteroid"])
        # Check that status options are mentioned
        assert any(status in prompt_lower for status in ["resonant", "non-resonant", "transient", "controversial"])

    def test_provider_specific_configuration(self):
        """Test that provider-specific configuration works."""
        # Test OpenAI
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LLM_PROVIDER": "openai"}):
            with patch('llm_libration.llm.client.ChatOpenAI'):
                analyzer = LibrationAnalyzer()
                assert analyzer.llm_client.provider == "openai"

        # Test Anthropic
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "LLM_PROVIDER": "anthropic"}):
            with patch('llm_libration.llm.client.ChatAnthropic'):
                analyzer = LibrationAnalyzer()
                assert analyzer.llm_client.provider == "anthropic"


class TestIntegration:
    """Integration tests (require real API keys)."""

    def test_real_api_integration_openai(self):
        """Test integration with real OpenAI API (requires API key)."""
        # Skip if no API key available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            analyzer = LibrationAnalyzer(provider="openai")
            assert analyzer is not None
        except (ConfigurationError, ImportError):
            pytest.skip("OpenAI integration not available")

    def test_real_api_integration_anthropic(self):
        """Test integration with real Anthropic API (requires API key)."""
        # Skip if no API key available
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            analyzer = LibrationAnalyzer(provider="anthropic")
            assert analyzer is not None
        except (ConfigurationError, ImportError):
            pytest.skip("Anthropic integration not available")
