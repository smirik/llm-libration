#!/usr/bin/env python3
import sys
from pathlib import Path
from llm_libration import LibrationAnalyzer

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python demo.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image file {image_path} not found")
        sys.exit(1)

    providers = ["openai", "anthropic", "openrouter", "ollama"]

    print(f"Analyzing image: {image_path}")
    print("-" * 40)

    for provider in providers:
        try:
            analyzer = LibrationAnalyzer(provider=provider)
            result = analyzer.analyze_image(image_path)
            print(f"{provider.upper()}: {result.value}")
        except Exception as e:
            print(f"{provider.upper()}: Error - {e}")
