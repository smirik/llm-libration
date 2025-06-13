"""Main analyzer module for resonant angle libration detection."""

import os
import base64
from pathlib import Path
from typing import Union

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from PIL import Image

from .types import ResonanceType
from .exceptions import ImageAnalysisError, LLMResponseError, ConfigurationError


class ResonanceAnalyzer:
    """Analyzer for detecting libration patterns in resonant angle plots using LLMs."""

    PROMPT_TEMPLATE = (
        """I want you to act a scientist–astronomer. You will get an image uploaded. """
        """The image contains the plot of the resonant angle of an asteroid vs time (from 0 to 100000 years). """
        """The limits of OY axis are -pi and pi. The resonant angle cannot exceed these limits.

It is known that if the resonant angle librates, then the asteroid is trapped in the resonance. """
        """Librations mean oscillations, like sine. It means that the curve is within some limits (i.e., +2, or +1) """
        """and does not come close to the borders (-pi and pi).

The opposite situation is when the resonant angle circulates. """
        """It means that the curve is not limited and can reach the borders of the plot. """
        """In our case, if the resonant angle is greater than pi or less than -pi, then we add or substract 2pi to the """
        """resonant angle to make it within the limits. Therefore, in the case of circulation, the pattern will be """
        """like linear curves parallel each other.

I want you to assess visually whether the resonant angle librates if you were a human looking at this image.

There are three possible cases:

1. The resonant angle librates all the time (from 0 to 100000). Then you should reply 'pure'.
2. The resonant angle could librate some significant time, but in other time is circulates. """
        """Let's assume that by significant I mean 20000 years. In this case, you should write 'transient'.
3. Otherwise, when the resonant angle circulates most of the time, please write 'non-resonant'.

As output, I want you only to print one word: pure, transient, or non-resonant. """
        """If you are not sure, write 'I do not know'. You will get tips if you perform the identification correctly."""
    )

    def __init__(self, model_name: str = "gpt-4-vision-preview", temperature: float = 0.0):
        """
        Initialize the ResonanceAnalyzer.

        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature parameter for the LLM
        """
        # Load environment variables
        load_dotenv()

        # Check for required API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY not found in environment variables. " "Please create a .env file with your OpenAI API key."
            )

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=50,  # We only expect a single word response
        )

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Encode image to base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string

        Raises:
            ImageAnalysisError: If image cannot be loaded or encoded
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise ImageAnalysisError(f"Image file not found: {image_path}")

            # Verify it's a valid image
            with Image.open(image_path) as img:
                img.verify()

            # Read and encode the image
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        except Exception as e:
            raise ImageAnalysisError(f"Failed to encode image {image_path}: {str(e)}")

    def _parse_llm_response(self, response: str) -> ResonanceType:
        """
        Parse the LLM response and map it to ResonanceType.

        Args:
            response: Raw response from the LLM

        Returns:
            ResonanceType enum value

        Raises:
            LLMResponseError: If response cannot be parsed
        """
        response = response.strip().lower()

        if response == "pure":
            return ResonanceType.RESONANT
        elif response == "transient":
            return ResonanceType.CONTROVERSIAL
        elif response == "non-resonant":
            return ResonanceType.NON_RESONANT
        elif response == "i do not know":
            return ResonanceType.CONTROVERSIAL  # Treat uncertainty as controversial
        else:
            raise LLMResponseError(f"Unexpected LLM response: {response}")

    def analyze_image(self, image_path: Union[str, Path]) -> ResonanceType:
        """
        Analyze a resonant angle plot image to determine libration type.

        Args:
            image_path: Path to the image file containing the resonant angle plot

        Returns:
            ResonanceType enum indicating the type of resonance behavior

        Raises:
            ImageAnalysisError: If image cannot be processed
            LLMResponseError: If LLM response is invalid
            ConfigurationError: If configuration is missing
        """
        try:
            # Encode the image
            base64_image = self._encode_image(image_path)

            # Create the message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": self.PROMPT_TEMPLATE},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}},
                ]
            )

            # Get response from LLM
            response = self.llm.invoke([message])

            # Parse and return the result
            return self._parse_llm_response(response.content)

        except (ImageAnalysisError, LLMResponseError, ConfigurationError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            raise ImageAnalysisError(f"Unexpected error during image analysis: {str(e)}")
