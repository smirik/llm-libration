# LLM Libration

A Python package for analyzing resonant angle libration patterns in astronomical data using Large Language Models (LLMs). The package supports multiple LLM providers including OpenAI, Anthropic, OpenRouter, and local models via Ollama.

## Features

-   🔍 **Automated Analysis**: AI-powered detection of libration patterns in resonant angle plots
-   🌐 **Multiple LLM Providers**: Support for OpenAI, Anthropic, OpenRouter, and Ollama
-   📊 **Plot Classification**: Categorizes resonance behavior as pure, transient, or non-resonant
-   🧪 **Easy Integration**: Simple Python API for astronomy research workflows
-   ⚙️ **Configurable**: Environment-based configuration for different providers and models
-   🔒 **Type Safety**: Full type hints and comprehensive error handling

## Installation

```bash
pip install llm-libration
```

### All Providers Included

All LLM providers are now included by default with the installation:

-   **OpenAI** (via `langchain-openai`)
-   **Anthropic** (via `langchain-anthropic`)
-   **OpenRouter** (via `langchain-openai`)
-   **Ollama** (via `ollama` + `langchain-community`)

No additional packages are required!

## Quick Start

### Basic Usage with OpenAI (default)

```python
from llm_libration import LibrationAnalyzer

# Initialize analyzer (uses OpenAI by default)
analyzer = LibrationAnalyzer()

# Analyze an image
result = analyzer.analyze_image("path/to/resonance_plot.png")
print(f"Resonance type: {result}")  # Output: ResonanceType.RESONANT
```

### Using Different Providers

```python
from llm_libration import LibrationAnalyzer

# Use Anthropic Claude
analyzer = LibrationAnalyzer(provider="anthropic")

# Use OpenRouter
analyzer = LibrationAnalyzer(provider="openrouter")

# Use local Ollama model
analyzer = LibrationAnalyzer(provider="ollama")

# Custom model for any provider
analyzer = LibrationAnalyzer(
    provider="anthropic",
    model_name="claude-sonnet-4"
)
```

## Configuration

The package uses environment variables for configuration. Create a `.env` file in your project root:

### OpenAI Configuration (Default)

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=openai/gpt-4.1  # Optional, this is the default
```

### Anthropic Configuration

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL_NAME=claude-sonnet-4  # Optional, this is the default
```

### OpenRouter Configuration

```bash
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL_NAME=anthropic/claude-sonnet-4  # Optional, this is the default
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # Optional, this is the default
```

### Ollama Configuration (Local Models)

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434  # Optional, this is the default
OLLAMA_MODEL_NAME=gemma3  # Optional, this is the default
```

**Available Ollama Vision Models:**

-   `gemma3` (recommended, latest model with excellent vision capabilities)
-   `llama3.2-vision` (good balance of performance and accuracy)
-   `llava` (lightweight, good for development)
-   `gemma2:2b-vision` (compact model)

**Note:** The package uses the `ollama` Python package for optimal vision support with automatic fallback handling.

### Advanced Configuration

```bash
# Custom prompt template (optional)
PROMPT_TEMPLATE="Your custom analysis prompt here..."
```

## Environment Setup

1. **Copy the example configuration:**

    ```bash
    cp .env.dist .env
    ```

2. **Edit `.env` with your credentials:**
   Choose your preferred provider and set the appropriate API key.

3. **For Ollama users:**
   Make sure Ollama is running locally:
    ```bash
    ollama serve
    ollama pull gemma3  # or your preferred vision model
    ```

## API Reference

### LibrationAnalyzer

```python
class LibrationAnalyzer:
    def __init__(self, model_name: str = None, provider: str = None):
        """
        Initialize the analyzer.

        Args:
            model_name: Override the default model name
            provider: Override the default provider (openai, anthropic, openrouter, ollama)
        """

    def analyze_image(self, image_path: Union[str, Path]) -> ResonanceType:
        """
        Analyze a resonance plot image.

        Args:
            image_path: Path to the image file

        Returns:
            ResonanceType enum (RESONANT, NON_RESONANT, or CONTROVERSIAL)
        """
```

### ResonanceType

```python
from enum import Enum

class ResonanceType(Enum):
    RESONANT = "resonant"        # Pure libration detected
    NON_RESONANT = "non-resonant" # Circulation detected
    CONTROVERSIAL = "controversial" # Transient or uncertain behavior
```

## Provider Comparison

| Provider       | Pros                                        | Cons                                  | Best For                                         |
| -------------- | ------------------------------------------- | ------------------------------------- | ------------------------------------------------ |
| **OpenAI**     | Excellent vision capabilities, fast         | Requires API key, paid service        | Production use, high accuracy                    |
| **Anthropic**  | Strong reasoning, good vision               | Requires API key, paid service        | Research, detailed analysis                      |
| **OpenRouter** | Access to many models, competitive pricing  | Requires API key, paid service        | Cost-effective access to multiple models         |
| **Ollama**     | Free, local, private, proper vision support | Requires local setup, model downloads | Development, privacy-sensitive work, offline use |

## Error Handling

The package provides specific exceptions for different error scenarios:

```python
from llm_libration.exceptions import (
    ImageAnalysisError,      # Image processing issues
    LLMResponseError,        # LLM response parsing issues
    ConfigurationError       # Missing or invalid configuration
)

try:
    result = analyzer.analyze_image("plot.png")
except ImageAnalysisError as e:
    print(f"Image processing failed: {e}")
except LLMResponseError as e:
    print(f"LLM response issue: {e}")
except ConfigurationError as e:
    print(f"Configuration problem: {e}")
```

## Backward Compatibility

The original `ResonanceAnalyzer` class name is still supported:

```python
from llm_libration import ResonanceAnalyzer  # Legacy name

analyzer = ResonanceAnalyzer()  # Works exactly the same
```

## Development

To set up for development:

```bash
git clone https://github.com/your-username/llm-libration.git
cd llm-libration
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{llm_libration,
  title={LLM Libration: AI-Powered Analysis of Resonant Angle Libration Patterns},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/llm-libration}
}
```
