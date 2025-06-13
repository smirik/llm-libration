"""Tests for the LLM Client class."""

import os
import pytest
from unittest.mock import Mock, patch
import tempfile
from PIL import Image

from llm_libration.llm import LLMClient
from llm_libration.exceptions import ImageAnalysisError, LLMResponseError, ConfigurationError


class TestLLMClient:
    """Test cases for LLMClient class."""

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

    def test_llm_client_init_openai(self):
        """Test LLMClient initialization with OpenAI provider."""
        with patch('llm_libration.llm.client.ChatOpenAI') as mock_chat:
            client = LLMClient(provider="openai", model_name="gpt-4", api_key="test-key")
            assert client is not None
            assert client.provider == "openai"
            assert client.model_name == "gpt-4"
            mock_chat.assert_called_once_with(model="gpt-4", temperature=0.0, max_tokens=50, api_key="test-key")

    def test_llm_client_init_anthropic(self):
        """Test LLMClient initialization with Anthropic provider."""
        with patch('llm_libration.llm.client.ChatAnthropic') as mock_chat:
            client = LLMClient(provider="anthropic", model_name="claude-sonnet-4", api_key="test-key")
            assert client is not None
            assert client.provider == "anthropic"
            mock_chat.assert_called_once_with(model="claude-sonnet-4", temperature=0.0, max_tokens=50, api_key="test-key")

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
                max_tokens=50,
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
            return LLMClient(provider="openai", model_name="openai/gpt-4.1", api_key="test-key")

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
        """Test successful image analysis with OpenAI provider."""
        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = "pure"
        openai_client.llm.invoke = Mock(return_value=mock_response)

        result = openai_client.analyze_image_with_prompt(sample_image, "test prompt")

        assert result == "pure"
        openai_client.llm.invoke.assert_called_once()

    def test_analyze_image_with_prompt_success_ollama(self, ollama_client, sample_image):
        """Test successful image analysis with Ollama using direct client."""
        # Mock the direct ollama response
        mock_response = {'message': {'content': 'pure'}}
        ollama_client.ollama_client.chat = Mock(return_value=mock_response)

        result = ollama_client.analyze_image_with_prompt(sample_image, "test prompt")

        assert result == "pure"
        ollama_client.ollama_client.chat.assert_called_once()

        # Verify the call was made with correct parameters
        call_args = ollama_client.ollama_client.chat.call_args
        assert call_args[1]['model'] == "gemma3"
        assert call_args[1]['messages'][0]['role'] == 'user'
        assert call_args[1]['messages'][0]['content'] == "test prompt"
        assert 'images' in call_args[1]['messages'][0]

    def test_analyze_image_with_prompt_ollama_fallback_to_base64(self, ollama_client, sample_image):
        """Test Ollama direct client falling back to base64 encoding."""
        # Mock the first call to fail (direct path), second to succeed (base64)
        mock_response = {'message': {'content': 'pure'}}

        # Create a side effect that raises on first call, succeeds on second
        def side_effect(*args, **kwargs):
            if 'images' in kwargs['messages'][0]:
                images = kwargs['messages'][0]['images']
                if images and not images[0].startswith('data:') and len(images[0]) < 200:
                    # This looks like a file path, simulate failure
                    raise Exception("File path not supported")
            return mock_response

        ollama_client.ollama_client.chat = Mock(side_effect=side_effect)

        result = ollama_client.analyze_image_with_prompt(sample_image, "test prompt")

        assert result == "pure"
        # Should be called twice - once with path, once with base64
        assert ollama_client.ollama_client.chat.call_count == 2

    def test_analyze_image_with_prompt_llm_error(self, openai_client, sample_image):
        """Test image analysis with LLM error."""
        # Mock the LLM to raise an exception
        openai_client.llm.invoke = Mock(side_effect=Exception("LLM API error"))

        with pytest.raises(LLMResponseError, match="LLM interaction failed"):
            openai_client.analyze_image_with_prompt(sample_image, "test prompt")

    def test_analyze_image_with_prompt_image_error(self, openai_client):
        """Test image analysis with image processing error."""
        with pytest.raises(ImageAnalysisError, match="Image file not found"):
            openai_client.analyze_image_with_prompt("nonexistent.png", "test prompt")

    def test_ollama_vision_analysis_failure(self, ollama_client, sample_image):
        """Test Ollama direct client when both path and base64 approaches fail."""
        # Mock both calls to fail
        ollama_client.ollama_client.chat = Mock(side_effect=Exception("Vision analysis failed"))

        with pytest.raises(LLMResponseError, match="Ollama vision analysis failed"):
            ollama_client.analyze_image_with_prompt(sample_image, "test prompt")

    def test_anthropic_client_message_format(self):
        """Test that Anthropic client uses correct message format."""
        with patch('llm_libration.llm.client.ChatAnthropic') as mock_anthropic:
            client = LLMClient(provider="anthropic", model_name="claude-sonnet-4", api_key="test-key")

            # Create a test image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img = Image.new('RGB', (100, 100), color='white')
                img.save(f.name, 'PNG')

                # Mock the response
                mock_response = Mock()
                mock_response.content = "pure"
                client.llm.invoke = Mock(return_value=mock_response)

                result = client.analyze_image_with_prompt(f.name, "test prompt")

                assert result == "pure"
                # Verify that invoke was called with HumanMessage format
                client.llm.invoke.assert_called_once()
                call_args = client.llm.invoke.call_args[0][0]
                assert len(call_args) == 1  # Should be a list with one HumanMessage

                # Cleanup
                os.unlink(f.name)
