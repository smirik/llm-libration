"""LLM client for handling image analysis and response processing."""

import base64
from pathlib import Path
from typing import Union

import ollama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain.schema import HumanMessage
from PIL import Image

from ..exceptions import ImageAnalysisError, LLMResponseError, ConfigurationError


class LLMClient:
    """Client for handling LLM interactions and image processing."""

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

        if self.provider == "openai":
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0.0,
                max_tokens=50,
                api_key=api_key,
            )
        elif self.provider == "anthropic":
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=0.0,
                max_tokens=50,
                api_key=api_key,
            )
        elif self.provider == "openrouter":
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0.0,
                max_tokens=50,
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

    def _analyze_with_ollama_direct(self, image_path: Union[str, Path], prompt: str) -> str:
        """
        Analyze image using direct ollama client for better vision support.

        Args:
            image_path: Path to the image file
            prompt: Analysis prompt

        Returns:
            Response from Ollama model
        """
        try:
            response = self.ollama_client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt, 'images': [str(image_path)]}],  # Direct path - ollama handles encoding
            )
            return response['message']['content']
        except Exception as e:
            try:
                base64_image, _ = self.encode_image(image_path)
                response = self.ollama_client.chat(
                    model=self.model_name, messages=[{'role': 'user', 'content': prompt, 'images': [base64_image]}]  # Base64 encoded image
                )
                return response['message']['content']
            except Exception as e2:
                raise LLMResponseError(f"Ollama vision analysis failed: {str(e2)}")

    def analyze_image_with_prompt(self, image_path: Union[str, Path], prompt: str) -> str:
        """
        Analyze an image using the provided prompt.

        Args:
            image_path: Path to the image file
            prompt: Text prompt for analysis

        Returns:
            Raw response from the LLM

        Raises:
            ImageAnalysisError: If image processing fails
            LLMResponseError: If LLM interaction fails
        """
        try:
            if self.provider == "ollama":
                return self._analyze_with_ollama_direct(image_path, prompt)
            else:
                base64_image, mime_type = self.encode_image(image_path)
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}", "detail": "high"}},
                    ]
                )
                response = self.llm.invoke([message])
                return response.content

        except ImageAnalysisError:
            raise
        except Exception as e:
            raise LLMResponseError(f"LLM interaction failed: {str(e)}")
