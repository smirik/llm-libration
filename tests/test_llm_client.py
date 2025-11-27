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
from llm_libration.types import ResonanceType


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
            mock_chat.assert_called_once_with(model="gpt-4", temperature=1.0, max_tokens=4000, api_key="test-key")

    def test_llm_client_init_anthropic(self):
        """Test LLMClient initialization with Anthropic provider."""
        with patch('llm_libration.llm.client.ChatAnthropic') as mock_chat:
            client = LLMClient(provider="anthropic", model_name="claude-sonnet-4", api_key="test-key")
            assert client is not None
            assert client.provider == "anthropic"
            assert client.model_name == "claude-sonnet-4"
            mock_chat.assert_called_once_with(model="claude-sonnet-4", temperature=1.0, max_tokens=4000, api_key="test-key")

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
                temperature=1.0,
                max_tokens=4000,
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
    def openrouter_client(self):
        """Create an OpenRouter LLMClient instance for testing."""
        with patch('llm_libration.llm.client.ChatOpenAI'):
            return LLMClient(
                provider="openrouter",
                model_name="anthropic/claude-sonnet-4",
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
            )

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

        mock_result = LibrationAnalysisResult(status=ResonanceType.RESONANT, subtype=ResonantSubtype.APOCENTRIC_LIBRATION)

        # Mock the LangChain structured output
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_result

        openai_client.llm.with_structured_output = Mock(return_value=mock_structured_llm)

        result = openai_client.analyze_image_with_prompt(sample_image, "test prompt")

        assert isinstance(result, LibrationAnalysisResult)
        assert result.status == ResonanceType.RESONANT
        assert result.subtype == ResonantSubtype.APOCENTRIC_LIBRATION
        openai_client.llm.with_structured_output.assert_called_once_with(LibrationAnalysisResult)

    def test_analyze_image_with_prompt_success_ollama(self, ollama_client, sample_image):
        """Test successful image analysis with Ollama using structured output."""
        # Mock the structured JSON response
        mock_response = {'message': {'content': '{"status": "resonant", "subtype": "apocentric libration"}'}}
        ollama_client.ollama_client.chat = Mock(return_value=mock_response)

        result = ollama_client.analyze_image_with_prompt(sample_image, "test prompt")

        assert isinstance(result, LibrationAnalysisResult)
        assert result.status == ResonanceType.RESONANT
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
        assert result.status == ResonanceType.NON_RESONANT
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

    def test_openrouter_fallback_to_json_cleanup(self, openrouter_client, sample_image):
        """Ensure OpenRouter falls back to manual JSON parsing when structured output fails."""
        mock_structured = Mock()
        mock_structured.invoke.side_effect = Exception("Invalid JSON response")
        openrouter_client.llm.with_structured_output = Mock(return_value=mock_structured)

        ai_message = Mock()
        ai_message.content = (
            "Looking at this resonant pattern, the behavior is stable.\n"
            "```json\n{\"status\": \"resonant\", \"subtype\": \"apocentric libration\"}\n```"
        )
        openrouter_client.llm.invoke = Mock(return_value=ai_message)

        result = openrouter_client.analyze_image_with_prompt(sample_image, "test prompt")

        assert result.status == ResonanceType.RESONANT
        assert result.subtype == "apocentric libration"
        assert openrouter_client.llm.invoke.called

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

                mock_result = LibrationAnalysisResult(status=ResonanceType.TRANSIENT, subtype=TransientSubtype.ALTERNATING)

                mock_structured_llm = Mock()
                mock_structured_llm.invoke.return_value = mock_result
                client.llm.with_structured_output = Mock(return_value=mock_structured_llm)

                result = client.analyze_image_with_prompt(f.name, "test prompt")

                assert isinstance(result, LibrationAnalysisResult)
                assert result.status == ResonanceType.TRANSIENT
                assert result.subtype == TransientSubtype.ALTERNATING
                client.llm.with_structured_output.assert_called_once_with(LibrationAnalysisResult)

                os.unlink(f.name)

    def test_structured_output_method(self, openai_client, sample_image):
        """Test the structured output method returns proper LibrationAnalysisResult."""
        # Mock the structured LLM result
        mock_result = LibrationAnalysisResult(status=ResonanceType.CONTROVERSIAL, subtype="unclear pattern")

        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_result
        openai_client.llm.with_structured_output = Mock(return_value=mock_structured_llm)

        result = openai_client.analyze_image_with_prompt(sample_image, "test prompt")

        assert isinstance(result, LibrationAnalysisResult)
        assert result.status == ResonanceType.CONTROVERSIAL
        assert result.subtype == "unclear pattern"

    def test_clean_json_response_handles_prose(self, ollama_client):
        """Ensure JSON cleaner extracts objects surrounded by prose."""
        messy_content = (
            "Looking at this resonant angle plot, here's the summary:\n"
            "Some commentary before the data.\n"
            "{\"status\": \"non-resonant\", \"subtype\": \"circulation\"}\n"
            "Misc text after."
        )
        cleaned = ollama_client._clean_json_response(messy_content)
        parsed = json.loads(cleaned)
        assert parsed["status"] == "non-resonant"
        assert parsed["subtype"] == "circulation"

    def test_llm_client_init_huggingface(self):
        """Test LLMClient initialization with HuggingFace provider."""
        with patch('llm_libration.llm.client.HF_AVAILABLE', True), \
             patch('llm_libration.llm.client.TORCH_AVAILABLE', True), \
             patch('llm_libration.llm.client.AutoProcessor') as mock_processor, \
             patch('llm_libration.llm.client.AutoModelForImageTextToText') as mock_model, \
             patch('llm_libration.llm.client.torch') as mock_torch:
            mock_torch.bfloat16 = 'bfloat16'
            mock_model.from_pretrained.return_value = Mock(device='cuda')

            client = LLMClient(
                provider="huggingface",
                model_name="Qwen/Qwen2-VL-2B-Instruct",
                quantization="none"
            )
            assert client is not None
            assert client.provider == "huggingface"
            assert client.model_name == "Qwen/Qwen2-VL-2B-Instruct"
            assert client.quantization == "none"
            mock_processor.from_pretrained.assert_called_once_with("Qwen/Qwen2-VL-2B-Instruct")

    def test_llm_client_init_huggingface_4bit(self):
        """Test LLMClient initialization with HuggingFace provider and 4-bit quantization."""
        with patch('llm_libration.llm.client.HF_AVAILABLE', True), \
             patch('llm_libration.llm.client.TORCH_AVAILABLE', True), \
             patch('llm_libration.llm.client.AutoProcessor') as mock_processor, \
             patch('llm_libration.llm.client.AutoModelForImageTextToText') as mock_model, \
             patch('llm_libration.llm.client.BitsAndBytesConfig') as mock_bnb, \
             patch('llm_libration.llm.client.torch') as mock_torch:
            mock_torch.bfloat16 = 'bfloat16'
            mock_model.from_pretrained.return_value = Mock(device='cuda')

            client = LLMClient(
                provider="huggingface",
                model_name="Qwen/Qwen2-VL-7B-Instruct",
                quantization="4bit"
            )
            assert client is not None
            assert client.provider == "huggingface"
            assert client.quantization == "4bit"
            mock_bnb.assert_called_once()
            call_kwargs = mock_bnb.call_args[1]
            assert call_kwargs['load_in_4bit'] is True

    def test_llm_client_init_huggingface_8bit(self):
        """Test LLMClient initialization with HuggingFace provider and 8-bit quantization."""
        with patch('llm_libration.llm.client.HF_AVAILABLE', True), \
             patch('llm_libration.llm.client.TORCH_AVAILABLE', True), \
             patch('llm_libration.llm.client.AutoProcessor') as mock_processor, \
             patch('llm_libration.llm.client.AutoModelForImageTextToText') as mock_model, \
             patch('llm_libration.llm.client.BitsAndBytesConfig') as mock_bnb, \
             patch('llm_libration.llm.client.torch') as mock_torch:
            mock_torch.bfloat16 = 'bfloat16'
            mock_model.from_pretrained.return_value = Mock(device='cuda')

            client = LLMClient(
                provider="huggingface",
                model_name="Qwen/Qwen2-VL-7B-Instruct",
                quantization="8bit"
            )
            assert client is not None
            assert client.quantization == "8bit"
            mock_bnb.assert_called_once_with(load_in_8bit=True)

    def test_llm_client_init_huggingface_missing_transformers(self):
        """Test HuggingFace provider fails gracefully when transformers not installed."""
        with patch('llm_libration.llm.client.HF_AVAILABLE', False):
            with pytest.raises(ConfigurationError, match="transformers.*not installed"):
                LLMClient(provider="huggingface", model_name="test-model")

    def test_llm_client_init_huggingface_missing_torch(self):
        """Test HuggingFace provider fails gracefully when torch not installed."""
        with patch('llm_libration.llm.client.HF_AVAILABLE', True), \
             patch('llm_libration.llm.client.TORCH_AVAILABLE', False):
            with pytest.raises(ConfigurationError, match="PyTorch"):
                LLMClient(provider="huggingface", model_name="test-model")

    def test_llm_client_init_mlx(self):
        """Test LLMClient initialization with MLX provider."""
        with patch('llm_libration.llm.client.MLX_AVAILABLE', True), \
             patch('llm_libration.llm.client.mlx_load') as mock_load, \
             patch('llm_libration.llm.client.mlx_load_config') as mock_config, \
             patch('llm_libration.llm.client.platform') as mock_platform:
            mock_platform.system.return_value = "Darwin"
            mock_platform.machine.return_value = "arm64"
            mock_load.return_value = (Mock(), Mock())
            mock_config.return_value = {}

            client = LLMClient(
                provider="mlx",
                model_name="mlx-community/Qwen2-VL-2B-Instruct-4bit"
            )
            assert client is not None
            assert client.provider == "mlx"
            assert client.model_name == "mlx-community/Qwen2-VL-2B-Instruct-4bit"
            mock_load.assert_called_once_with("mlx-community/Qwen2-VL-2B-Instruct-4bit")

    def test_llm_client_init_mlx_missing_mlx_vlm(self):
        """Test MLX provider fails gracefully when mlx-vlm not installed."""
        with patch('llm_libration.llm.client.MLX_AVAILABLE', False):
            with pytest.raises(ConfigurationError, match="mlx-vlm.*not installed"):
                LLMClient(provider="mlx", model_name="test-model")

    def test_mlx_platform_validation_linux(self):
        """Test MLX provider rejects non-macOS platforms."""
        with patch('llm_libration.llm.client.MLX_AVAILABLE', True), \
             patch('llm_libration.llm.client.platform') as mock_platform:
            mock_platform.system.return_value = "Linux"
            with pytest.raises(ConfigurationError, match="only available on macOS"):
                LLMClient(provider="mlx", model_name="test-model")

    def test_mlx_platform_validation_intel_mac(self):
        """Test MLX provider rejects Intel Macs."""
        with patch('llm_libration.llm.client.MLX_AVAILABLE', True), \
             patch('llm_libration.llm.client.platform') as mock_platform:
            mock_platform.system.return_value = "Darwin"
            mock_platform.machine.return_value = "x86_64"
            with pytest.raises(ConfigurationError, match="Apple Silicon"):
                LLMClient(provider="mlx", model_name="test-model")

    @pytest.fixture
    def huggingface_client(self):
        """Create a HuggingFace LLMClient instance for testing."""
        with patch('llm_libration.llm.client.HF_AVAILABLE', True), \
             patch('llm_libration.llm.client.TORCH_AVAILABLE', True), \
             patch('llm_libration.llm.client.AutoProcessor') as mock_processor, \
             patch('llm_libration.llm.client.AutoModelForImageTextToText') as mock_model, \
             patch('llm_libration.llm.client.torch') as mock_torch:
            mock_torch.bfloat16 = 'bfloat16'
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()
            mock_model_instance = Mock()
            mock_model_instance.device = 'cuda'
            mock_model.from_pretrained.return_value = mock_model_instance

            client = LLMClient(
                provider="huggingface",
                model_name="Qwen/Qwen2-VL-2B-Instruct",
                quantization="none"
            )
            client._mock_torch = mock_torch
            return client

    @pytest.fixture
    def mlx_client(self):
        """Create an MLX LLMClient instance for testing."""
        with patch('llm_libration.llm.client.MLX_AVAILABLE', True), \
             patch('llm_libration.llm.client.mlx_load') as mock_load, \
             patch('llm_libration.llm.client.mlx_load_config') as mock_config, \
             patch('llm_libration.llm.client.platform') as mock_platform:
            mock_platform.system.return_value = "Darwin"
            mock_platform.machine.return_value = "arm64"
            mock_load.return_value = (Mock(), Mock())
            mock_config.return_value = {}

            return LLMClient(
                provider="mlx",
                model_name="mlx-community/Qwen2-VL-2B-Instruct-4bit"
            )

    def test_analyze_image_with_prompt_huggingface_error(self):
        """Test HuggingFace provider handles errors gracefully."""
        with patch('llm_libration.llm.client.HF_AVAILABLE', True), \
             patch('llm_libration.llm.client.TORCH_AVAILABLE', True), \
             patch('llm_libration.llm.client.AutoProcessor') as mock_processor, \
             patch('llm_libration.llm.client.AutoModelForImageTextToText') as mock_model, \
             patch('llm_libration.llm.client.torch') as mock_torch:
            mock_torch.bfloat16 = 'bfloat16'
            mock_model.from_pretrained.return_value = Mock(device='cuda')

            client = LLMClient(
                provider="huggingface",
                model_name="Qwen/Qwen2-VL-2B-Instruct",
                quantization="none"
            )

            client.hf_processor.apply_chat_template.side_effect = Exception("Model error")

            with pytest.raises(LLMResponseError, match="HuggingFace analysis failed"):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    img = Image.new('RGB', (100, 100), color='red')
                    img.save(f.name, 'PNG')
                    try:
                        client.analyze_image_with_prompt(f.name, "test prompt")
                    finally:
                        os.unlink(f.name)

    def test_analyze_image_with_prompt_success_mlx(self, mlx_client, sample_image):
        """Test successful image analysis with MLX provider."""
        with patch('llm_libration.llm.client.mlx_apply_chat_template') as mock_template, \
             patch('llm_libration.llm.client.mlx_generate') as mock_generate:
            mock_template.return_value = "formatted prompt"
            mock_generate.return_value = '{"status": "non-resonant", "subtype": "circulation"}'

            result = mlx_client.analyze_image_with_prompt(sample_image, "test prompt")

            assert isinstance(result, LibrationAnalysisResult)
            assert result.status == ResonanceType.NON_RESONANT
            assert result.subtype == "circulation"

    def test_analyze_image_with_prompt_mlx_image_not_found(self, mlx_client):
        """Test MLX provider handles missing image file."""
        with pytest.raises(ImageAnalysisError, match="Image file not found"):
            mlx_client.analyze_image_with_prompt("nonexistent.png", "test prompt")
