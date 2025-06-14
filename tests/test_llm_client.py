"""Tests for LLM client functionality."""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from llm_libration.exceptions import ConfigurationError, ImageAnalysisError, LLMResponseError
from llm_libration.llm.client import LLMClient
from llm_libration.llm.schema import LibrationAnalysisResult


class TestLLMClient:
    """Test cases for LLMClient."""

    @pytest.fixture
    def sample_image(self):
        """Create a temporary test image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(f.name, 'PNG')
            yield f.name
        os.unlink(f.name)

    def test_llm_client_init_openai(self):
        """Test LLMClient initialization with OpenAI provider."""
        with patch('llm_libration.llm.client.ChatOpenAI') as mock_chat:
            client = LLMClient(provider="openai", model_name="gpt-4", api_key="test-key")
            assert client is not None
            assert client.provider == "openai"
            assert client.model_name == "gpt-4"
            mock_chat.assert_called_once_with(model="gpt-4", temperature=0.0, max_tokens=2000, api_key="test-key")

    def test_llm_client_init_anthropic(self):
        """Test LLMClient initialization with Anthropic provider."""
        with patch('llm_libration.llm.client.ChatAnthropic') as mock_chat:
            client = LLMClient(provider="anthropic", model_name="claude-sonnet-4", api_key="test-key")
            assert client is not None
            assert client.provider == "anthropic"
            assert client.model_name == "claude-sonnet-4"
            mock_chat.assert_called_once_with(model="claude-sonnet-4", temperature=0.0, max_tokens=2000, api_key="test-key")

    def test_llm_client_init_openrouter(self):
        """Test LLMClient initialization with OpenRouter provider."""
        with patch('llm_libration.llm.client.ChatOpenAI') as mock_chat:
            client = LLMClient(
                provider="openrouter", model_name="anthropic/claude-sonnet-4", api_key="test-key", base_url="https://openrouter.ai/api/v1"
            )
            assert client is not None
            assert client.provider == "openrouter"
            mock_chat.assert_called_once_with(
                model="anthropic/claude-sonnet-4",
                temperature=0.0,
                max_tokens=2000,
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
            )

    def test_llm_client_init_ollama(self):
        """Test LLMClient initialization with Ollama provider."""
        with patch('llm_libration.llm.client.ollama') as mock_ollama:
            client = LLMClient(provider="ollama", model_name="gemma3", base_url="http://localhost:11434")
            assert client is not None
            assert client.provider == "ollama"
            assert client.ollama_client == mock_ollama.Client.return_value

    def test_llm_client_init_unsupported_provider(self):
        """Test LLMClient initialization fails with unsupported provider."""
        with pytest.raises(ConfigurationError, match="Unsupported provider: unsupported"):
            LLMClient(provider="unsupported", model_name="some-model", api_key="test-key")

    @pytest.fixture
    def openai_client(self):
        """Create an OpenAI LLMClient instance for testing."""
        with patch('llm_libration.llm.client.ChatOpenAI'):
            return LLMClient(provider="openai", model_name="gpt-4", api_key="test-key")

    @pytest.fixture
    def ollama_client(self):
        """Create an Ollama LLMClient instance for testing."""
        with patch('llm_libration.llm.client.ollama') as mock_ollama:
            client = LLMClient(provider="ollama", model_name="gemma3", base_url="http://localhost:11434")
            client.ollama_client = mock_ollama
            return client

    def test_encode_image_success(self, openai_client, sample_image):
        """Test successful image encoding."""
        result = openai_client.encode_image(sample_image)
        assert isinstance(result, tuple)
        assert len(result) == 2
        base64_string, mime_type = result
        assert isinstance(base64_string, str)
        assert isinstance(mime_type, str)
        assert len(base64_string) > 0
        assert mime_type.startswith('image/')

    def test_encode_image_nonexistent_file(self, openai_client):
        """Test encoding of non-existent image file."""
        with pytest.raises(ImageAnalysisError, match="Image file not found"):
            openai_client.encode_image("nonexistent.png")

    def test_encode_image_invalid_file(self, openai_client):
        """Test encoding of invalid image file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not an image")
            f.flush()

            with pytest.raises(ImageAnalysisError, match="Failed to encode image"):
                openai_client.encode_image(f.name)

            os.unlink(f.name)

    def test_analyze_image_with_prompt_success_openai(self, openai_client, sample_image):
        """Test successful image analysis with OpenAI provider using structured output."""
        # Mock the structured LLM result
        from llm_libration.llm.schema import ResonantSubtype

        mock_result = LibrationAnalysisResult(status="resonant", subtype=ResonantSubtype.APOCENTRIC_LIBRATION)

        # Mock the LangChain structured output
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_result

        openai_client.llm.with_structured_output = Mock(return_value=mock_structured_llm)

        result = openai_client.analyze_image_with_prompt(sample_image, "test prompt")

        assert isinstance(result, LibrationAnalysisResult)
        assert result.status == "resonant"
        assert result.subtype == ResonantSubtype.APOCENTRIC_LIBRATION
        openai_client.llm.with_structured_output.assert_called_once_with(LibrationAnalysisResult)

    def test_analyze_image_with_prompt_success_ollama(self, ollama_client, sample_image):
        """Test successful image analysis with Ollama using structured output."""
        # Mock the structured JSON response
        mock_response = {'message': {'content': '{"status": "resonant", "subtype": "apocentric libration"}'}}
        ollama_client.ollama_client.chat = Mock(return_value=mock_response)

        result = ollama_client.analyze_image_with_prompt(sample_image, "test prompt")

        assert isinstance(result, LibrationAnalysisResult)
        assert result.status == "resonant"
        assert result.subtype == "apocentric libration"  # This is a string, not enum - that's fine for JSON parsing
        ollama_client.ollama_client.chat.assert_called()

    def test_analyze_image_with_prompt_ollama_fallback_to_base64(self, ollama_client, sample_image):
        """Test Ollama client falling back to base64 encoding."""
        # Mock responses for different fallback attempts
        mock_response = {'message': {'content': '{"status": "non-resonant", "subtype": "circulation"}'}}

        def side_effect(*args, **kwargs):
            # Simulate first calls failing, last succeeding
            if hasattr(side_effect, 'call_count'):
                side_effect.call_count += 1
            else:
                side_effect.call_count = 1

            if side_effect.call_count < 3:
                raise Exception("Structured format not supported")
            return mock_response

        ollama_client.ollama_client.chat = Mock(side_effect=side_effect)

        result = ollama_client.analyze_image_with_prompt(sample_image, "test prompt")

        assert isinstance(result, LibrationAnalysisResult)
        assert result.status == "non-resonant"
        assert result.subtype == "circulation"
        assert ollama_client.ollama_client.chat.call_count == 3

    def test_analyze_image_with_prompt_llm_error(self, openai_client, sample_image):
        """Test image analysis with LLM error."""
        # Mock the LangChain structured output to raise an exception
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.side_effect = Exception("API error")
        openai_client.llm.with_structured_output = Mock(return_value=mock_structured_llm)

        with pytest.raises(LLMResponseError, match="Openai structured analysis failed"):
            openai_client.analyze_image_with_prompt(sample_image, "test prompt")

    def test_analyze_image_with_prompt_image_error(self, openai_client):
        """Test image analysis with image processing error."""
        with pytest.raises(ImageAnalysisError, match="Image file not found"):
            openai_client.analyze_image_with_prompt("nonexistent.png", "test prompt")

    def test_ollama_vision_analysis_failure(self, ollama_client, sample_image):
        """Test Ollama client when all approaches fail."""
        # Mock all calls to fail
        ollama_client.ollama_client.chat = Mock(side_effect=Exception("Vision analysis failed"))

        with pytest.raises(LLMResponseError, match="Ollama structured analysis failed"):
            ollama_client.analyze_image_with_prompt(sample_image, "test prompt")

    def test_anthropic_client_structured_output(self):
        """Test that Anthropic client uses LangChain structured output."""
        with patch('llm_libration.llm.client.ChatAnthropic'):
            client = LLMClient(provider="anthropic", model_name="claude-sonnet-4", api_key="test-key")

            # Create a test image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img = Image.new('RGB', (100, 100), color='white')
                img.save(f.name, 'PNG')

                # Mock the structured LLM result
                from llm_libration.llm.schema import TransientSubtype

                mock_result = LibrationAnalysisResult(status="transient", subtype=TransientSubtype.ALTERNATING)

                mock_structured_llm = Mock()
                mock_structured_llm.invoke.return_value = mock_result
                client.llm.with_structured_output = Mock(return_value=mock_structured_llm)

                result = client.analyze_image_with_prompt(f.name, "test prompt")

                assert isinstance(result, LibrationAnalysisResult)
                assert result.status == "transient"
                assert result.subtype == TransientSubtype.ALTERNATING
                client.llm.with_structured_output.assert_called_once_with(LibrationAnalysisResult)

                os.unlink(f.name)

    def test_structured_output_method(self, openai_client, sample_image):
        """Test the structured output method returns proper LibrationAnalysisResult."""
        # Mock the structured LLM result
        mock_result = LibrationAnalysisResult(status="controversial", subtype="unclear pattern")

        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_result
        openai_client.llm.with_structured_output = Mock(return_value=mock_structured_llm)

        result = openai_client.analyze_image_with_prompt(sample_image, "test prompt")

        assert isinstance(result, LibrationAnalysisResult)
        assert result.status == "controversial"
        assert result.subtype == "unclear pattern"
