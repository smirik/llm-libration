#!/usr/bin/env python3
"""
Example usage of the llm-libration package.

This script demonstrates how to use the ResonanceAnalyzer to analyze
resonant angle plots and determine libration behavior.
"""

import sys
from pathlib import Path

from llm_libration import ResonanceAnalyzer, ResonanceType
from llm_libration.exceptions import ImageAnalysisError, ConfigurationError


def main():
    """Main example function."""
    print("LLM-Libration Example")
    print("=" * 30)

    # Check if image path is provided
    if len(sys.argv) != 2:
        print("Usage: python example.py <path_to_image>")
        print("Example: python example.py /path/to/resonant_angle_plot.png")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    try:
        # Initialize the analyzer
        print("Initializing ResonanceAnalyzer...")
        analyzer = ResonanceAnalyzer()

        # Analyze the image
        print(f"Analyzing image: {image_path}")
        result = analyzer.analyze_image(image_path)

        # Display results
        print("\nAnalysis Results:")
        print("-" * 20)
        print(f"Resonance Type: {result}")

        # Provide interpretation
        if result == ResonanceType.RESONANT:
            print("Interpretation: Pure libration detected - the asteroid is trapped in resonance")
            print("               with oscillatory behavior within bounds.")
        elif result == ResonanceType.NON_RESONANT:
            print("Interpretation: Circulation behavior detected - the resonant angle")
            print("               reaches plot boundaries, indicating no resonance trapping.")
        elif result == ResonanceType.CONTROVERSIAL:
            print("Interpretation: Transient or uncertain behavior detected - mixed")
            print("               libration/circulation or unclear from the analysis.")

    except ConfigurationError as e:
        print(f"Configuration Error: {e}")
        print("Make sure you have created a .env file with your OpenAI API key.")
        sys.exit(1)

    except ImageAnalysisError as e:
        print(f"Image Analysis Error: {e}")
        print("Please check that the image file exists and is a valid image format.")
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
