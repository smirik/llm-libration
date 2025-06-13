"""LLM client for handling image analysis and response processing."""

import base64
import json
from pathlib import Path
from typing import Union

import ollama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage
from PIL import Image

from .schema import LibrationAnalysisResult
from ..exceptions import ImageAnalysisError, LLMResponseError, ConfigurationError


class LLMClient:
    """Client for handling LLM interactions and image processing with structured outputs."""

    def __init__(self, provider: str, model_name: str, api_key: str = "", base_url: str = ""):
        """
        Initialize the LLM client.

        Args:
            provider: LLM provider (openai, anthropic, openrouter, ollama)
            model_name: Name of the model to use
            api_key: API key for the provider (not needed for Ollama)
            base_url: Base URL for the provider (needed for OpenRouter and Ollama)
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key

        # Increase max_tokens for structured outputs
        max_tokens = 2000

        if self.provider == "openai":
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0.0,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        elif self.provider == "anthropic":
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=0.0,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        elif self.provider == "openrouter":
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0.0,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=base_url,
            )
        elif self.provider == "ollama":
            self.ollama_client = ollama.Client(host=base_url)
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

    def _analyze_with_ollama_structured(self, image_path: Union[str, Path], prompt: str) -> LibrationAnalysisResult:
        """
        Analyze image using Ollama with structured outputs.

        Args:
            image_path: Path to the image file
            prompt: Analysis prompt

        Returns:
            Structured libration analysis result
        """
        try:
            # Enhanced prompt for structured output
            json_prompt = f"""{prompt}

Please respond with a JSON object that matches this exact schema:
{{
  "status": "resonant" | "non-resonant" | "transient" | "controversial",
  "subtype": "string describing the specific type like 'apocentric libration', 'circulation', etc."
}}

Return only valid JSON, no additional text."""

            try:
                # Try using structured output format if supported
                response = self.ollama_client.chat(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': json_prompt, 'images': [str(image_path)]}],
                    format=LibrationAnalysisResult.get_ollama_schema(),
                )
                content = response['message']['content']
                parsed_data = json.loads(content)
                return LibrationAnalysisResult(**parsed_data)
            except Exception:
                # Fallback to regular JSON mode
                try:
                    response = self.ollama_client.chat(
                        model=self.model_name,
                        messages=[{'role': 'user', 'content': json_prompt, 'images': [str(image_path)]}],
                    )
                    content = response['message']['content']

                    # Clean up response
                    content = content.strip()
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                    content = content.strip()

                    parsed_data = json.loads(content)
                    return LibrationAnalysisResult(**parsed_data)
                except Exception:
                    # Try with base64 encoded image
                    base64_image, _ = self.encode_image(image_path)
                    response = self.ollama_client.chat(
                        model=self.model_name,
                        messages=[{'role': 'user', 'content': json_prompt, 'images': [base64_image]}],
                    )
                    content = response['message']['content']

                    # Clean up response
                    content = content.strip()
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                    content = content.strip()

                    parsed_data = json.loads(content)
                    return LibrationAnalysisResult(**parsed_data)

        except Exception as e:
            raise LLMResponseError(f"Ollama structured analysis failed: {str(e)}")

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
            if self.provider in ["openai", "anthropic", "openrouter"]:
                return self._analyze_with_langchain_structured(image_path, prompt)
            elif self.provider == "ollama":
                return self._analyze_with_ollama_structured(image_path, prompt)
            else:
                raise ConfigurationError(f"Unsupported provider: {self.provider}")

        except (ImageAnalysisError, LLMResponseError, ConfigurationError):
            raise
        except Exception as e:
            raise LLMResponseError(f"Unexpected error during structured analysis: {str(e)}")
