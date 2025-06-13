#!/usr/bin/env python3
"""
Example usage of the LLM Libration package with multiple providers.

This script demonstrates how to analyze resonant angle plots using different
LLM providers (OpenAI, Anthropic, OpenRouter, Ollama).
"""

import sys
from pathlib import Path

from llm_libration import LibrationAnalyzer, ResonanceType
from llm_libration.exceptions import ImageAnalysisError, LLMResponseError, ConfigurationError


def analyze_with_provider(image_path: str, provider: str, model_name: str = None):
    """Analyze an image using a specific provider."""
    try:
        print(f"\n🔍 Analyzing with {provider.upper()}...")

        if model_name:
            analyzer = LibrationAnalyzer(provider=provider, model_name=model_name)
            print(f"   Model: {model_name}")
        else:
            analyzer = LibrationAnalyzer(provider=provider)

        result = analyzer.analyze_image(image_path)

        # Format the result nicely
        result_descriptions = {
            ResonanceType.RESONANT: "🟢 Pure libration detected (resonant behavior)",
            ResonanceType.NON_RESONANT: "🔴 Circulation detected (non-resonant behavior)",
            ResonanceType.CONTROVERSIAL: "🟡 Transient/uncertain behavior detected",
        }

        print(f"   Result: {result_descriptions.get(result, str(result))}")
        return result

    except ConfigurationError as e:
        print(f"   ❌ Configuration error: {e}")
        return None
    except ImageAnalysisError as e:
        print(f"   ❌ Image analysis error: {e}")
        return None
    except LLMResponseError as e:
        print(f"   ❌ LLM response error: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return None


def main():
    """Main example function."""
    if len(sys.argv) != 2:
        print("Usage: python example.py <path_to_image>")
        print("\nExample:")
        print("  python example.py /path/to/resonance_plot.png")
        sys.exit(1)

    image_path = sys.argv[1]

    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)

    print("🚀 LLM Libration Analysis - Multi-Provider Example")
    print("=" * 55)
    print(f"📁 Analyzing image: {image_path}")

    # Example 1: OpenAI (default provider)
    print("\n📊 Example 1: Using OpenAI (default)")
    print("-" * 40)
    analyze_with_provider(image_path, "openai")

    # Example 2: Anthropic Claude
    print("\n📊 Example 2: Using Anthropic Claude")
    print("-" * 40)
    analyze_with_provider(image_path, "anthropic")

    # Example 3: OpenRouter
    print("\n📊 Example 3: Using OpenRouter")
    print("-" * 40)
    analyze_with_provider(image_path, "openrouter")

    # Example 4: Ollama (local model)
    print("\n📊 Example 4: Using Ollama (local)")
    print("-" * 40)
    print("   💡 Now with proper vision support!")
    print("   📝 Available models: llama3.2-vision, llava, gemma2:2b-vision")
    analyze_with_provider(image_path, "ollama")

    # Example 5: Custom model with specific provider
    print("\n📊 Example 5: Custom Models")
    print("-" * 40)
    analyze_with_provider(image_path, "anthropic", "claude-3-opus-20240229")
    analyze_with_provider(image_path, "openai", "gpt-4o")
    analyze_with_provider(image_path, "ollama", "llava")  # Alternative Ollama model

    # Example 6: Demonstrate backwards compatibility
    print("\n📊 Example 6: Backwards Compatibility")
    print("-" * 40)
    try:
        from llm_libration import ResonanceAnalyzer  # Legacy name

        print("🔍 Using legacy ResonanceAnalyzer class...")
        analyzer = ResonanceAnalyzer()  # Uses default OpenAI
        result = analyzer.analyze_image(image_path)
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   ❌ Error with legacy interface: {e}")

    print("\n✨ Analysis complete!")
    print("\n💡 Tips:")
    print("   • Set LLM_PROVIDER in your .env file to change the default provider")
    print("   • Different providers may give different results for the same image")
    print("   • OpenAI and Anthropic typically provide the most reliable results")
    print("   • Ollama is free but requires local setup and a vision-capable model")
    print("   • Use provider-specific API keys in your .env file")


if __name__ == "__main__":
    main()
