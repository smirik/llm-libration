"""Tests for the LibrationAnalyzer class."""

import os
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
from PIL import Image

from llm_libration.analyzer import LibrationAnalyzer
from llm_libration.types import ResonanceType
from llm_libration.exceptions import ImageAnalysisError, LLMResponseError, ConfigurationError


class TestLibrationAnalyzer:
    """Test cases for LibrationAnalyzer class."""

    @pytest.fixture
    def mock_openai_env(self):
        """Mock OpenAI environment variables."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}):
            yield

    @pytest.fixture
    def mock_anthropic_env(self):
        """Mock Anthropic environment variables."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"}):
            yield

    @pytest.fixture
    def mock_ollama_env(self):
        """Mock Ollama environment variables."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "OLLAMA_BASE_URL": "http://localhost:11434"}):
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
        with patch('llm_libration.llm.client.Ollama'):
            return LibrationAnalyzer()

    def test_init_with_openai_api_key(self, mock_openai_env):
        """Test successful initialization with OpenAI provider."""
        with patch('llm_libration.llm.client.ChatOpenAI') as mock_chat:
            analyzer = LibrationAnalyzer()
            assert analyzer is not None
            assert analyzer.llm_client is not None
            mock_chat.assert_called_once()

    def test_init_with_anthropic_api_key(self, mock_anthropic_env):
        """Test successful initialization with Anthropic provider."""
        with patch('llm_libration.llm.client.ChatAnthropic') as mock_chat:
            analyzer = LibrationAnalyzer()
            assert analyzer is not None
            assert analyzer.llm_client is not None
            mock_chat.assert_called_once()

    def test_init_with_ollama(self, mock_ollama_env):
        """Test successful initialization with Ollama provider."""
        with patch('llm_libration.llm.client.ollama') as mock_ollama:
            analyzer = LibrationAnalyzer()
            assert analyzer is not None
            assert analyzer.llm_client is not None
            # Verify the direct ollama client is set up
            assert analyzer.llm_client.ollama_client == mock_ollama.Client.return_value

    def test_init_without_api_key_openai(self):
        """Test initialization fails without OpenAI API key."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            with pytest.raises(ConfigurationError, match="OPENAI_API_KEY not found"):
                LibrationAnalyzer()

    def test_init_without_api_key_anthropic(self):
        """Test initialization fails without Anthropic API key."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}, clear=True):
            with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY not found"):
                LibrationAnalyzer()

    def test_init_custom_model_and_provider(self, mock_openai_env):
        """Test initialization with custom model and provider."""
        with patch('llm_libration.llm.client.ChatOpenAI') as mock_chat:
            LibrationAnalyzer(model_name="gpt-4", provider="openai")
            mock_chat.assert_called_once()
            # Check that the custom model was passed to LLMClient
            call_args = mock_chat.call_args
            assert call_args[1]['model'] == "gpt-4"

    def test_init_provider_override(self, mock_openai_env):
        """Test that provider parameter overrides config."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch('llm_libration.llm.client.ChatAnthropic') as mock_chat:
                # Override the default OpenAI provider with Anthropic
                LibrationAnalyzer(provider="anthropic")
                mock_chat.assert_called_once()

    def test_init_unsupported_provider(self, mock_openai_env):
        """Test initialization fails with unsupported provider."""
        with pytest.raises(ConfigurationError, match="Unsupported provider"):
            LibrationAnalyzer(provider="unsupported")

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
    def test_parse_llm_response_valid(self, openai_analyzer, llm_response, expected_type):
        """Test parsing of valid LLM responses."""
        result = openai_analyzer._parse_llm_response(llm_response)
        assert result == expected_type

    def test_parse_llm_response_invalid(self, openai_analyzer):
        """Test parsing of invalid LLM response."""
        with pytest.raises(LLMResponseError, match="Unexpected LLM response"):
            openai_analyzer._parse_llm_response("invalid response")

    def test_analyze_image_success_openai(self, openai_analyzer, sample_image):
        """Test successful image analysis with OpenAI."""
        # Mock the LLM client response
        openai_analyzer.llm_client.analyze_image_with_prompt = Mock(return_value="pure")

        result = openai_analyzer.analyze_image(sample_image)

        assert result == ResonanceType.RESONANT
        openai_analyzer.llm_client.analyze_image_with_prompt.assert_called_once()

    def test_analyze_image_success_anthropic(self, anthropic_analyzer, sample_image):
        """Test successful image analysis with Anthropic."""
        # Mock the LLM client response
        anthropic_analyzer.llm_client.analyze_image_with_prompt = Mock(return_value="transient")

        result = anthropic_analyzer.analyze_image(sample_image)

        assert result == ResonanceType.CONTROVERSIAL
        anthropic_analyzer.llm_client.analyze_image_with_prompt.assert_called_once()

    def test_analyze_image_success_ollama(self, ollama_analyzer, sample_image):
        """Test successful image analysis with Ollama."""
        # Mock the LLM client response (now with proper vision support)
        ollama_analyzer.llm_client.analyze_image_with_prompt = Mock(return_value="non-resonant")

        result = ollama_analyzer.analyze_image(sample_image)

        assert result == ResonanceType.NON_RESONANT
        ollama_analyzer.llm_client.analyze_image_with_prompt.assert_called_once()

    def test_analyze_image_llm_error(self, openai_analyzer, sample_image):
        """Test image analysis with LLM error."""
        # Mock the LLM client to raise an exception
        openai_analyzer.llm_client.analyze_image_with_prompt = Mock(side_effect=LLMResponseError("LLM error"))

        with pytest.raises(LLMResponseError):
            openai_analyzer.analyze_image(sample_image)

    def test_analyze_image_invalid_response(self, openai_analyzer, sample_image):
        """Test image analysis with invalid LLM response."""
        # Mock the LLM client response with invalid content
        openai_analyzer.llm_client.analyze_image_with_prompt = Mock(return_value="invalid")

        with pytest.raises(LLMResponseError):
            openai_analyzer.analyze_image(sample_image)

    def test_analyze_image_pathlib_path(self, openai_analyzer, sample_image):
        """Test image analysis with pathlib.Path input."""
        openai_analyzer.llm_client.analyze_image_with_prompt = Mock(return_value="transient")

        result = openai_analyzer.analyze_image(Path(sample_image))

        assert result == ResonanceType.CONTROVERSIAL

    def test_prompt_template_access(self, openai_analyzer):
        """Test that the prompt template is accessible through config."""
        from llm_libration import config

        assert isinstance(config.prompt_template, str)
        assert len(config.prompt_template) > 100  # Should be a substantial prompt
        assert "astronomer" in config.prompt_template.lower()
        assert "pure" in config.prompt_template.lower()
        assert "transient" in config.prompt_template.lower()
        assert "non-resonant" in config.prompt_template.lower()

    def test_provider_specific_configuration(self):
        """Test that provider-specific configuration is used correctly."""
        # Test OpenAI configuration
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "openai-key", "OPENAI_MODEL_NAME": "gpt-4"}):
            with patch('llm_libration.llm.client.ChatOpenAI') as mock_client:
                LibrationAnalyzer()
                mock_client.assert_called_once()
                call_kwargs = mock_client.call_args[1]
                assert call_kwargs['api_key'] == "openai-key"
                assert call_kwargs['model'] == "gpt-4"

        # Test Anthropic configuration
        with patch.dict(
            os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "anthropic-key", "ANTHROPIC_MODEL_NAME": "claude-sonnet-4"}
        ):
            with patch('llm_libration.llm.client.ChatAnthropic') as mock_client:
                LibrationAnalyzer()
                mock_client.assert_called_once()
                call_kwargs = mock_client.call_args[1]
                assert call_kwargs['api_key'] == "anthropic-key"
                assert call_kwargs['model'] == "claude-sonnet-4"


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
    def test_real_api_integration_openai(self):
        """Test with real OpenAI API (requires API key)."""
        # This test only runs if OPENAI_API_KEY is available
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new('RGB', (400, 300), color='white')
            img.save(f.name, 'PNG')

            try:
                with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
                    analyzer = LibrationAnalyzer()
                    # This might fail due to the simple test image, but shouldn't crash
                    result = analyzer.analyze_image(f.name)
                    assert isinstance(result, ResonanceType)
            except (LLMResponseError, ImageAnalysisError):
                # These are acceptable for a simple test image
                pass
            finally:
                os.unlink(f.name)

    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Requires ANTHROPIC_API_KEY environment variable")
    def test_real_api_integration_anthropic(self):
        """Test with real Anthropic API (requires API key)."""
        # This test only runs if ANTHROPIC_API_KEY is available
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new('RGB', (400, 300), color='white')
            img.save(f.name, 'PNG')

            try:
                with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}):
                    analyzer = LibrationAnalyzer()
                    # This might fail due to the simple test image, but shouldn't crash
                    result = analyzer.analyze_image(f.name)
                    assert isinstance(result, ResonanceType)
            except (LLMResponseError, ImageAnalysisError):
                # These are acceptable for a simple test image
                pass
            finally:
                os.unlink(f.name)
