"""LLM client for handling image analysis and response processing."""

import base64
import json
import logging
import platform
import re
from pathlib import Path
from typing import Union

try:
    import ollama  # type: ignore
except ImportError:  # pragma: no cover - fallback when ollama package is absent
    ollama = None

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ChatAnthropic = None

try:
    from langchain.schema import HumanMessage
except ImportError:  # pragma: no cover - optional dependency
    HumanMessage = None

# HuggingFace Transformers imports
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig  # type: ignore
    HF_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    HF_AVAILABLE = False
    AutoModelForImageTextToText = None
    AutoProcessor = None
    BitsAndBytesConfig = None

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False
    torch = None

# MLX imports (macOS Apple Silicon only)
try:
    from mlx_vlm import load as mlx_load, generate as mlx_generate  # type: ignore
    from mlx_vlm.prompt_utils import apply_chat_template as mlx_apply_chat_template  # type: ignore
    MLX_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    MLX_AVAILABLE = False
    mlx_load = None
    mlx_generate = None
    mlx_apply_chat_template = None

from PIL import Image

try:
    from json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    JSON_REPAIR_AVAILABLE = False
    repair_json = None

from .schema import LibrationAnalysisResult
from ..exceptions import ImageAnalysisError, LLMResponseError, ConfigurationError
from ..config import config

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for handling LLM interactions and image processing with structured outputs."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: str = "",
        base_url: str = "",
        quantization: str = "none",
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
    ):
        """
        Initialize the LLM client.

        Args:
            provider: LLM provider (openai, anthropic, openrouter, ollama, huggingface, mlx)
            model_name: Name of the model to use
            api_key: API key for the provider (not needed for Ollama/HuggingFace/MLX)
            base_url: Base URL for the provider (needed for OpenRouter and Ollama)
            quantization: Quantization mode for HuggingFace (none, 4bit, 8bit)
            device_map: Device mapping strategy for HuggingFace
            torch_dtype: Torch dtype for HuggingFace model loading
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key

        # Increase max_tokens for structured outputs while keeping headroom for reasoning
        # Set to 2000 to avoid truncation errors with reasoning models
        max_tokens = 4000
        temperature = 1.0  # gpt-5-mini and similar structured APIs only support the default temperature

        if self.provider == "openai":
            if ChatOpenAI is None:
                raise ConfigurationError(
                    "OpenAI provider requested but 'langchain-openai' is not installed. " "Install it with `pip install langchain-openai`."
                )
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        elif self.provider == "anthropic":
            if ChatAnthropic is None:
                raise ConfigurationError(
                    "Anthropic provider requested but 'langchain-anthropic' is not installed. "
                    "Install it with `pip install langchain-anthropic`."
                )
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        elif self.provider == "openrouter":
            if ChatOpenAI is None:
                raise ConfigurationError(
                    "OpenRouter provider requested but 'langchain-openai' is not installed. "
                    "Install it with `pip install langchain-openai`."
                )
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=base_url,
            )
        elif self.provider == "ollama":
            if ollama is None:
                raise ConfigurationError(
                    "Ollama provider requested but the 'ollama' Python package is not installed. "
                    "Install it with `pip install ollama` to enable local vision models."
                )
            self.ollama_client = ollama.Client(host=base_url)
        elif self.provider == "huggingface":
            if not HF_AVAILABLE:
                raise ConfigurationError(
                    "HuggingFace provider requested but 'transformers' is not installed. "
                    "Install with: pip install transformers accelerate"
                )
            if not TORCH_AVAILABLE:
                raise ConfigurationError(
                    "HuggingFace provider requires PyTorch. Install with: pip install torch"
                )
            self.quantization = quantization
            self.device_map = device_map
            self.torch_dtype_str = torch_dtype
            self.hf_model, self.hf_processor = self._init_huggingface_model(model_name)
        elif self.provider == "mlx":
            if not MLX_AVAILABLE:
                raise ConfigurationError(
                    "MLX provider requested but 'mlx-vlm' is not installed. "
                    "Install with: pip install mlx-vlm"
                )
            self._validate_mlx_platform()
            self.mlx_model, self.mlx_processor = self._init_mlx_model(model_name)
            self.mlx_config = self.mlx_model.config  # Use model.config instead of load_config()
        else:
            raise ConfigurationError(f"Unsupported provider: {provider}")

    def encode_image(self, image_path: Union[str, Path]) -> tuple[str, str]:
        """
        Encode image to base64 string and detect format.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (base64_string, mime_type)

        Raises:
            ImageAnalysisError: If image cannot be loaded or encoded
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise ImageAnalysisError(f"Image file not found: {image_path}")

            with Image.open(image_path) as img:
                img_format = img.format.lower()
                img.verify()

            format_to_mime = {'jpeg': 'image/jpeg', 'jpg': 'image/jpeg', 'png': 'image/png', 'gif': 'image/gif', 'webp': 'image/webp'}
            mime_type = format_to_mime.get(img_format, 'image/jpeg')

            with open(image_path, "rb") as image_file:
                base64_string = base64.b64encode(image_file.read()).decode('utf-8')
                return base64_string, mime_type

        except Exception as e:
            raise ImageAnalysisError(f"Failed to encode image {image_path}: {str(e)}")

    def _analyze_with_langchain_structured(self, image_path: Union[str, Path], prompt: str) -> LibrationAnalysisResult:
        """
        Analyze image using LangChain's unified structured output for OpenAI, Anthropic, and OpenRouter.

        Args:
            image_path: Path to the image file
            prompt: Analysis prompt

        Returns:
            Structured libration analysis result
        """
        try:
            if HumanMessage is None:
                raise ConfigurationError("LangChain core is not installed. Install `langchain` to enable structured outputs.")
            base64_image, mime_type = self.encode_image(image_path)

            # Create structured LLM
            structured_llm = self.llm.with_structured_output(LibrationAnalysisResult)

            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}", "detail": "high"}},
                ]
            )

            result = structured_llm.invoke([message])
            return result

        except ImageAnalysisError:
            raise
        except Exception as e:
            raise LLMResponseError(f"{self.provider.title()} structured analysis failed: {str(e)}")

    def _analyze_with_openrouter_structured(self, image_path: Union[str, Path], prompt: str) -> LibrationAnalysisResult:
        """
        Analyze image using OpenRouter, falling back to manual JSON parsing when needed.
        """
        try:
            return self._analyze_with_langchain_structured(image_path, prompt)
        except ImageAnalysisError:
            raise
        except LLMResponseError as structured_error:
            logger.debug("OpenRouter structured output failed, attempting JSON fallback: %s", structured_error)
            return self._analyze_with_openrouter_json(image_path, prompt, structured_error)

    def _analyze_with_openrouter_json(
        self,
        image_path: Union[str, Path],
        prompt: str,
        structured_error: LLMResponseError | None = None,
    ) -> LibrationAnalysisResult:
        """
        Fallback path for OpenRouter models that ignore LangChain structured outputs.
        """
        try:
            if HumanMessage is None:
                raise ConfigurationError("LangChain core is not installed. Install `langchain` to enable structured outputs.")

            formatted_prompt = self._format_prompt_for_json(prompt)
            base64_image, mime_type = self.encode_image(image_path)

            message = HumanMessage(
                content=[
                    {"type": "text", "text": formatted_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}", "detail": "high"}},
                ]
            )

            response = self.llm.invoke([message])
            content = self._extract_text_content(response)
            cleaned_content = self._clean_json_response(content)
            parsed_data = json.loads(cleaned_content)

            if 'status' not in parsed_data or 'subtype' not in parsed_data:
                raise ValueError(f"Missing required fields in response: {parsed_data}")

            return LibrationAnalysisResult(**parsed_data)
        except Exception as exc:
            context = f"; structured attempt error: {structured_error}" if structured_error else ""
            raise LLMResponseError(f"OpenRouter JSON fallback failed to parse response: {exc}{context}")

    def _analyze_with_ollama_structured(self, image_path: Union[str, Path], prompt: str) -> LibrationAnalysisResult:
        """
        Analyze image using Ollama with structured outputs.

        Args:
            image_path: Path to the image file
            prompt: Analysis prompt (ignored for Ollama, uses config template)

        Returns:
            Structured libration analysis result
        """
        # Use the Ollama-specific prompt template from configuration
        direct_prompt = config.ollama_prompt_template

        last_error = None
        attempts = []

        # Attempt 1: Try regular JSON mode first (skip complex structured format)
        try:
            logger.debug(f"Ollama attempt 1: Simple JSON mode with image path: {image_path}")
            response = self.ollama_client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': direct_prompt, 'images': [str(image_path)]}],
            )
            content = response['message']['content']
            logger.debug(f"Ollama simple response: {content}")

            # Clean up response more carefully
            cleaned_content = self._clean_json_response(content)
            logger.debug(f"Cleaned content: {cleaned_content}")

            parsed_data = json.loads(cleaned_content)

            # Validate the parsed data has required fields
            if 'status' not in parsed_data or 'subtype' not in parsed_data:
                raise ValueError(f"Missing required fields in response: {parsed_data}")

            result = LibrationAnalysisResult(**parsed_data)
            logger.debug(f"Successfully parsed simple result: {result}")
            return result
        except Exception as e:
            last_error = e
            attempts.append(f"Simple JSON mode failed: {str(e)}")
            logger.debug(f"Ollama simple JSON mode failed: {e}")

        # Attempt 2: Try with base64 encoded image
        try:
            logger.debug(f"Ollama attempt 2: Base64 encoded image")
            base64_image, _ = self.encode_image(image_path)
            response = self.ollama_client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': direct_prompt, 'images': [base64_image]}],
            )
            content = response['message']['content']
            logger.debug(f"Ollama base64 response: {content}")

            # Clean up response more carefully
            cleaned_content = self._clean_json_response(content)
            logger.debug(f"Cleaned content: {cleaned_content}")

            parsed_data = json.loads(cleaned_content)

            # Validate the parsed data has required fields
            if 'status' not in parsed_data or 'subtype' not in parsed_data:
                raise ValueError(f"Missing required fields in response: {parsed_data}")

            result = LibrationAnalysisResult(**parsed_data)
            logger.debug(f"Successfully parsed base64 result: {result}")
            return result
        except Exception as e:
            last_error = e
            attempts.append(f"Base64 encoding failed: {str(e)}")
            logger.debug(f"Ollama base64 approach failed: {e}")

        # Attempt 3: Try with structured output format as last resort
        try:
            logger.debug(f"Ollama attempt 3: Structured format fallback with image path: {image_path}")
            response = self.ollama_client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': direct_prompt, 'images': [str(image_path)]}],
                format=LibrationAnalysisResult.get_ollama_schema(),
            )
            content = response['message']['content']
            logger.debug(f"Ollama structured fallback response: {content}")
            parsed_data = json.loads(content)
            result = LibrationAnalysisResult(**parsed_data)
            logger.debug(f"Successfully parsed structured fallback result: {result}")
            return result
        except Exception as e:
            last_error = e
            attempts.append(f"Structured format fallback failed: {str(e)}")
            logger.debug(f"Ollama structured format fallback failed: {e}")

        # All attempts failed, provide detailed error information
        error_details = "; ".join(attempts)
        raise LLMResponseError(
            f"Ollama structured analysis failed after all attempts. Details: {error_details}. Last error: {str(last_error)}"
        )

    def _resolve_torch_dtype(self, dtype_str: str):
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    def _init_huggingface_model(self, model_name: str):
        """Initialize HuggingFace vision-language model with optional quantization."""
        logger.info(f"Loading HuggingFace model: {model_name} (quantization={self.quantization})")

        quantization_config = None
        if self.quantization == "4bit":
            if BitsAndBytesConfig is None:
                raise ConfigurationError(
                    "4-bit quantization requires bitsandbytes. "
                    "Install with: pip install bitsandbytes"
                )
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=config.hf_4bit_quant_type,
                bnb_4bit_use_double_quant=config.hf_4bit_use_double_quant,
                bnb_4bit_compute_dtype=self._resolve_torch_dtype(config.hf_4bit_compute_dtype),
            )
        elif self.quantization == "8bit":
            if BitsAndBytesConfig is None:
                raise ConfigurationError(
                    "8-bit quantization requires bitsandbytes. "
                    "Install with: pip install bitsandbytes"
                )
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        processor = AutoProcessor.from_pretrained(model_name)

        model_kwargs = {
            "device_map": self.device_map,
            "torch_dtype": self._resolve_torch_dtype(self.torch_dtype_str),
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)

        logger.info(f"HuggingFace model loaded successfully on device: {model.device}")
        return model, processor

    @staticmethod
    def _validate_mlx_platform():
        """Validate that we're running on Apple Silicon macOS."""
        if platform.system() != "Darwin":
            raise ConfigurationError(
                f"MLX provider is only available on macOS. Current platform: {platform.system()}"
            )
        if platform.machine() not in ("arm64", "aarch64"):
            raise ConfigurationError(
                f"MLX provider requires Apple Silicon (M1/M2/M3/M4). Current architecture: {platform.machine()}"
            )

    def _init_mlx_model(self, model_name: str):
        """Initialize MLX vision-language model."""
        logger.info(f"Loading MLX model: {model_name}")
        try:
            model, processor = mlx_load(model_name)
            logger.info("MLX model loaded successfully")
            return model, processor
        except Exception as e:
            raise ConfigurationError(f"Failed to load MLX model '{model_name}': {str(e)}")

    def _analyze_with_huggingface(self, image_path: Union[str, Path], prompt: str) -> LibrationAnalysisResult:
        """Analyze image using HuggingFace vision-language model."""
        try:
            image = Image.open(image_path)
            formatted_prompt = self._format_prompt_for_json(prompt)

            model_type = getattr(self.hf_model.config, "model_type", "") or ""
            internvl_like = model_type.startswith("internvl")

            if internvl_like:
                # Ensure image placeholder is present for InternVL family.
                start_img = getattr(self.hf_processor, "image_token", None) or getattr(self.hf_processor.tokenizer, "start_image_token", "<img>")
                end_img = getattr(self.hf_processor, "end_image_token", None) or getattr(self.hf_processor.tokenizer, "end_image_token", "</img>")
                # Some processors expect image_token attributes; set explicitly.
                if hasattr(self.hf_processor, "image_token"):
                    self.hf_processor.image_token = start_img
                if hasattr(self.hf_processor, "end_image_token"):
                    self.hf_processor.end_image_token = end_img
                if start_img and start_img not in formatted_prompt:
                    closing = f"{end_img}\n" if end_img else ""
                    formatted_prompt = f"{start_img}{closing}{formatted_prompt}"
                # InternVL family expects explicit text/images instead of chat templates.
                inputs = self.hf_processor(
                    images=image,
                    text=formatted_prompt,
                    return_tensors="pt",
                )
            else:
                try:
                    # Most chat-style processors (e.g., LLaVA/Qwen VL) accept messages + images split
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": formatted_prompt},
                            ],
                        }
                    ]

                    inputs = self.hf_processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )

                    image_inputs = self.hf_processor(images=image, return_tensors="pt")
                    inputs.update(image_inputs)
                except Exception:
                    # Fallback for processors that do not support chat templates
                    inputs = self.hf_processor(
                        images=image,
                        text=formatted_prompt,
                        return_tensors="pt",
                    )

            inputs = {k: v.to(self.hf_model.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=config.hf_max_new_tokens,
                    do_sample=False,
                )

            input_len = inputs["input_ids"].shape[1]
            output_text = self.hf_processor.decode(
                generated_ids[0, input_len:],
                skip_special_tokens=True
            )

            logger.debug(f"HuggingFace raw output: {output_text}")

            cleaned_content = self._clean_json_response(output_text)
            parsed_data = json.loads(cleaned_content)

            if 'status' not in parsed_data or 'subtype' not in parsed_data:
                raise ValueError(f"Missing required fields in response: {parsed_data}")

            return LibrationAnalysisResult(**parsed_data)

        except ImageAnalysisError:
            raise
        except Exception as e:
            raise LLMResponseError(f"HuggingFace analysis failed: {str(e)}")

    def _analyze_with_mlx(self, image_path: Union[str, Path], prompt: str) -> LibrationAnalysisResult:
        """Analyze image using MLX vision-language model."""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise ImageAnalysisError(f"Image file not found: {image_path}")

            formatted_prompt = self._format_prompt_for_json(prompt)

            chat_prompt = mlx_apply_chat_template(
                self.mlx_processor,
                self.mlx_config,
                formatted_prompt,
                num_images=1
            )

            output = mlx_generate(
                self.mlx_model,
                self.mlx_processor,
                chat_prompt,
                [str(image_path)],
                max_tokens=config.mlx_max_tokens,
                temp=config.mlx_temperature,
                verbose=config.mlx_verbose,
            )

            # mlx_generate returns GenerationResult object with .text attribute
            output_text = output.text if hasattr(output, 'text') else (output if isinstance(output, str) else str(output))
            logger.debug(f"MLX raw output: {output_text}")

            cleaned_content = self._clean_json_response(output_text)
            parsed_data = json.loads(cleaned_content)

            if 'status' not in parsed_data or 'subtype' not in parsed_data:
                raise ValueError(f"Missing required fields in response: {parsed_data}")

            return LibrationAnalysisResult(**parsed_data)

        except ImageAnalysisError:
            raise
        except Exception as e:
            raise LLMResponseError(f"MLX analysis failed: {str(e)}")

    def _clean_json_response(self, content: str) -> str:
        """
        Extract and sanitize JSON content from an LLM response.

        Args:
            content: Raw response content from LLM

        Returns:
            Cleaned JSON string
        """
        if not content:
            raise ValueError("Empty response content")

        content = content.strip()
        logger.debug(f"Original content: {repr(content)}")

        # Prefer fenced code blocks if available
        code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
        if code_blocks:
            content = code_blocks[0].strip()
            logger.debug(f"Using fenced code block: {repr(content)}")
        else:
            # Remove stray fence markers
            content = content.replace("```json", "").replace("```", "").strip()

        # Simple case: already clean JSON object/array
        if (content.startswith('{') and content.endswith('}')) or (content.startswith('[') and content.endswith(']')):
            return content

        # Attempt to locate JSON substring by decoding at every brace
        decoder = json.JSONDecoder()
        for idx, char in enumerate(content):
            if char not in '{[':
                continue
            try:
                obj, _ = decoder.raw_decode(content[idx:])
                serialized = json.dumps(obj)
                logger.debug(f"Extracted JSON substring: {serialized}")
                return serialized
            except json.JSONDecodeError:
                continue

        # Fallback: try json_repair if available
        if JSON_REPAIR_AVAILABLE and repair_json is not None:
            try:
                repaired = repair_json(content)
                logger.debug(f"Repaired JSON using json_repair: {repaired}")
                # Validate the repaired JSON
                json.loads(repaired)
                return repaired
            except Exception as repair_error:
                logger.debug(f"json_repair failed: {repair_error}")

        raise ValueError(f"Content doesn't contain valid JSON: {repr(content)}")

    @staticmethod
    def _format_prompt_for_json(prompt: str) -> str:
        """Append explicit JSON-only instructions to the prompt."""
        prompt = prompt.rstrip()
        instructions = (
            "Respond ONLY with a single JSON object using double quotes and lowercase keys exactly as shown:\n"
            '{ "status": "<resonant|non-resonant|transient|controversial>", "subtype": "<short description>" }\n'
            "Do not include markdown fences or explanations."
        )
        return f"{prompt}\n\n{instructions}"

    @staticmethod
    def _extract_text_content(response) -> str:
        """
        Convert LangChain AIMessage content into plain text.
        """
        content = getattr(response, "content", response)

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for chunk in content:
                if isinstance(chunk, str):
                    parts.append(chunk)
                elif isinstance(chunk, dict):
                    text_value = chunk.get("text") or chunk.get("content")
                    if text_value:
                        parts.append(text_value)
            if parts:
                return "\n".join(parts)

        return str(content)

    def analyze_image_with_prompt(self, image_path: Union[str, Path], prompt: str) -> LibrationAnalysisResult:
        """
        Analyze an image using the provided prompt and return structured result.

        Args:
            image_path: Path to the image file
            prompt: Text prompt for analysis

        Returns:
            Structured libration analysis result

        Raises:
            ImageAnalysisError: If image processing fails
            LLMResponseError: If LLM interaction fails
        """
        try:
            if self.provider in ["openai", "anthropic"]:
                return self._analyze_with_langchain_structured(image_path, prompt)
            elif self.provider == "openrouter":
                return self._analyze_with_openrouter_structured(image_path, prompt)
            elif self.provider == "ollama":
                return self._analyze_with_ollama_structured(image_path, prompt)
            elif self.provider == "huggingface":
                return self._analyze_with_huggingface(image_path, prompt)
            elif self.provider == "mlx":
                return self._analyze_with_mlx(image_path, prompt)
            else:
                raise ConfigurationError(f"Unsupported provider: {self.provider}")

        except (ImageAnalysisError, LLMResponseError, ConfigurationError):
            raise
        except Exception as e:
            raise LLMResponseError(f"Unexpected error during structured analysis: {str(e)}")
