# LLM Libration

A Python package for analyzing resonant angle libration patterns in astronomical data using Large Language Models (LLMs). The package supports multiple LLM providers including OpenAI, Anthropic, OpenRouter, and local models via Ollama.

## Features

- 🔍 **Automated Analysis**: AI-powered detection of libration patterns in resonant angle plots
- 🌐 **Multiple LLM Providers**: Support for OpenAI, Anthropic, OpenRouter, and Ollama
- 📊 **Plot Classification**: Categorizes resonance behavior as pure, transient, or non-resonant
- 🧪 **Easy Integration**: Simple Python API for astronomy research workflows
- ⚙️ **Configurable**: Environment-based configuration for different providers and models
- 🔒 **Type Safety**: Full type hints and comprehensive error handling

## Installation

```bash
pip install llm-libration
```

### All Providers Included

All LLM providers are now included by default with the installation:

- **OpenAI** (via `langchain-openai`)
- **Anthropic** (via `langchain-anthropic`)
- **OpenRouter** (via `langchain-openai`)
- **Ollama** (via `ollama` + `langchain-community`)

No additional packages are required!

## Quick Start

### Command Line Usage

```bash
# Create a plot from CSV data
llm-libration plot input/463.csv

# Analyze an image with OpenAI (default)
llm-libration run input/demo.png

# Test all providers
llm-libration run input/demo.png --provider all
```

### Basic Python API Usage with OpenAI (default)

```python
from llm_libration import LibrationAnalyzer

# Initialize analyzer (uses OpenAI by default)
analyzer = LibrationAnalyzer()

# Analyze an image - returns detailed structured result
result = analyzer.analyze_image("path/to/resonance_plot.png")
print(f"Status: {result.status}")  # Output: resonant
print(f"Subtype: {result.subtype}")  # Output: apocentric libration
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

- `gemma3` (recommended, latest model with excellent vision capabilities)
- `llama3.2-vision` (good balance of performance and accuracy)
- `llava` (lightweight, good for development)
- `gemma2:2b-vision` (compact model)

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

## Command Line Interface

After installation, you can use `llm-libration` from the command line:

### Create Plots from CSV Data

```bash
# Basic plot creation with default parameters (single file)
llm-libration plot input/463.csv

# Process all CSV files in a folder recursively
llm-libration plot input/test

# Custom plot with specific columns and output (single file only)
llm-libration plot data.csv --x-column time --y-column resonance_angle --output-file my_plot.png

# Custom y-axis range (works for both files and folders)
llm-libration plot data.csv --y-min -3.14 --y-max 9.42
```

**Plot Command Options:**

- `INPUT_PATH`: Path to CSV file or folder containing CSV files (required)
- `--x-column`: Column for x-axis data (default: `times`)
- `--y-column`: Column for y-axis data (default: `angle`)
- `--output-file`: Output PNG path (only for single file input, ignored for folders)
- `--y-min`: Minimum y-axis value (default: `0`)
- `--y-max`: Maximum y-axis value (default: `2π`)

**Folder Processing:**
When a folder is provided as input, the command will:

- Recursively find all `.csv` files in the folder and subfolders
- Create plots for each CSV file found
- Save PNG files in the same directories as their corresponding CSV files
- Display progress and summary information

### Analyze Images

```bash
# Analyze with default provider (OpenAI)
llm-libration run input/demo.png

# Analyze with specific provider
llm-libration run input/demo.png --provider anthropic

# Analyze multiple images
llm-libration run input/demo.png input/463.png --provider openai

# Test all providers at once
llm-libration run input/demo.png --provider all

# Use custom model
llm-libration run input/demo.png --provider anthropic --model claude-sonnet-4
```

**Run Command Options:**

- `IMAGE_FILES`: One or more image paths (required)
- `--provider`: LLM provider (`openai`, `anthropic`, `openrouter`, `ollama`, `all`) (default: `openai`)
- `--model`: Custom model name (optional)

### Benchmark Evaluation

Evaluate LLM performance on categorized datasets:

```bash
# Run benchmark with default provider (OpenAI)
llm-libration benchmark input/benchmark

# Run benchmark with specific provider
llm-libration benchmark input/benchmark --provider anthropic

# Run benchmark with custom model
llm-libration benchmark input/benchmark --provider openai --model gpt-4-vision-preview
```

**Benchmark Command Options:**

- `BENCHMARK_DIR`: Path to directory with categorized subdirectories (required)
- `--provider`: LLM provider (`openai`, `anthropic`, `openrouter`, `ollama`) (default: `openai`)
- `--model`: Custom model name (optional)

**Benchmark Directory Structure:**
The benchmark directory should contain categorized subdirectories:

- `libration/` - Images showing resonant (libration) behavior
- `circulation/` or `non-resonant/` - Images showing non-resonant (circulation) behavior
- `transient/` - Images showing transient behavior (mapped to controversial)
- `controversial/` - Images that are difficult to classify

The command will:

- Recursively find all PNG files in these directories
- Analyze each image and compare with expected results
- Calculate classification metrics (Accuracy, Precision, Recall, F1 Score)
- Save detailed results in CSV and JSON formats
- Display comprehensive performance summary

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

    def analyze_image(self, image_path: Union[str, Path]) -> LibrationAnalysisResult:
        """
        Analyze a resonance plot image and return detailed structured result.

        Args:
            image_path: Path to the image file

        Returns:
            LibrationAnalysisResult with status and subtype information
        """

    def get_resonance_type(self, image_path: Union[str, Path]) -> ResonanceType:
        """
        Analyze a resonance plot image and return the ResonanceType enum.

        Args:
            image_path: Path to the image file

        Returns:
            ResonanceType enum (RESONANT, NON_RESONANT, or CONTROVERSIAL)
        """
```

### Plotting Functions

```python
from llm_libration.data.plot import create_plot, create_plots_from_input

# Single file plotting
def create_plot(
    input_file: Union[str, Path],
    x_column: str = 'times',
    y_column: str = 'angle',
    output_file: Optional[Union[str, Path]] = None,
    y_min: float = 0,
    y_max: float = 2 * np.pi
) -> str:
    """Create a plot from a single CSV file."""

# Unified plotting interface (files or folders)
def create_plots_from_input(
    input_path: Union[str, Path],
    x_column: str = 'times',
    y_column: str = 'angle',
    output_file: Optional[Union[str, Path]] = None,
    y_min: float = 0,
    y_max: float = 2 * np.pi
) -> Union[str, List[str]]:
    """Create plot(s) from CSV file(s). Returns single path for files, list for folders."""
```

### LibrationAnalysisResult

```python
from llm_libration.llm.schema import (
    LibrationAnalysisResult,
    ResonantSubtype,
    NonResonantSubtype,
    TransientSubtype
)

class LibrationAnalysisResult:
    status: str    # "resonant", "non-resonant", "transient", or "controversial"
    subtype: Union[ResonantSubtype, NonResonantSubtype, TransientSubtype, str]
    # Union type: enum for known subtypes, string for controversial cases
```

### Subtype Enums

The package provides specific enums for different categories of behavior:

**ResonantSubtype** (for status="resonant"):

- `CLEAR_LIBRATION` = "clear libration"
- `APOCENTRIC_LIBRATION` = "apocentric libration"
- `HIGH_AMPLITUDE_LIBRATION` = "high amplitude libration"
- `LONG_PERIOD_LIBRATION` = "long period libration"
- `NOISY_LIBRATION` = "noisy libration"
- `DOUBLE_LIBRATION` = "double libration"

**NonResonantSubtype** (for status="non-resonant"):

- `CIRCULATION` = "circulation"
- `SPARSE_CIRCULATION` = "sparse circulation"
- `CHAOTIC_CIRCULATION` = "chaotic circulation"
- `MIXED_CIRCULATION` = "mixed circulation"

**TransientSubtype** (for status="transient"):

- `NOISY_LIBRATION_TO_CIRCULATION` = "noisy libration to circulation"
- `APOCENTRIC_WITH_CIRCULATION` = "apocentric libration with circulation phases"
- `ALTERNATING` = "alternating libration and circulation"

```python
# Example usage
result = analyzer.analyze_image("plot.png")

# Type-safe access
if result.status == "resonant":
    if result.subtype == ResonantSubtype.APOCENTRIC_LIBRATION:
        print("Detected apocentric libration pattern")

# String comparison still works
if result.subtype == "circulation":
    print("Simple circulation detected")
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
    print(f"Status: {result.status}, Subtype: {result.subtype}")
except ImageAnalysisError as e:
    print(f"Image processing failed: {e}")
except LLMResponseError as e:
    print(f"LLM response issue: {e}")
except ConfigurationError as e:
    print(f"Configuration problem: {e}")
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
