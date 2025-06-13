# LLM-Libration

Identify librations of resonant angles by LLMs.

This Python package uses Large Language Models (LLMs) to analyze astronomical resonant angle plots and determine whether the angles show libration patterns, circulation patterns, or transient behavior.

## Installation

### From Source

```bash
git clone https://github.com/smirik/llm-libration.git
cd llm-libration
make setup
```

### Development Installation

```bash
git clone https://github.com/smirik/llm-libration.git
cd llm-libration
make install
source .venv/bin/activate
```

## Configuration

1. Copy the environment template:

```bash
cp .env.dist .env
```

2. Edit `.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_actual_api_key_here
```

You can get your API key from [OpenAI Platform](https://platform.openai.com/api-keys).

## Usage

### Basic Usage

```python
from llm_libration import ResonanceAnalyzer, ResonanceType

# Initialize the analyzer
analyzer = ResonanceAnalyzer()

# Analyze an image
result = analyzer.analyze_image("path/to/resonant_angle_plot.png")

print(f"Analysis result: {result}")
# Output: Analysis result: resonant
```

### Advanced Usage

```python
from llm_libration import ResonanceAnalyzer

# Use custom model and temperature
analyzer = ResonanceAnalyzer(
    model_name="gpt-4",
    temperature=0.1
)

result = analyzer.analyze_image("path/to/plot.png")

# Handle different result types
if result == ResonanceType.RESONANT:
    print("Pure libration detected")
elif result == ResonanceType.NON_RESONANT:
    print("Circulation behavior detected")
elif result == ResonanceType.CONTROVERSIAL:
    print("Transient or uncertain behavior detected")
```

### Command Line Usage

```bash
# Run the example script
make example IMAGE=path/to/your/image.png

# Or activate the environment and run directly
source .venv/bin/activate
python example.py path/to/your/image.png
```

## Expected Input

The package expects images containing plots of resonant angles vs time with the following characteristics:

-   **X-axis**: Time (typically 0 to 100,000 years)
-   **Y-axis**: Resonant angle (limits: -π to π)
-   **Content**: Plot showing the evolution of resonant angles over time

## Output Types

The analysis returns one of three `ResonanceType` values:

-   **`RESONANT`**: Pure libration (oscillatory behavior within bounds)
-   **`NON_RESONANT`**: Circulation (reaches plot boundaries)
-   **`CONTROVERSIAL`**: Transient behavior (mixed libration/circulation) or uncertain cases

## Development

This project uses a Makefile for common development tasks. Run `make help` to see all available commands.

### Quick Start

```bash
# Complete project setup
make setup

# Show all available commands
make help

# Run all code quality checks
make check

# Development workflow
make dev
```

### Common Commands

```bash
# Testing
make test                    # Run tests
make test-verbose           # Run tests with verbose output
make coverage               # Run tests with coverage report
make quick-test             # Run tests without coverage (faster)

# Code Quality
make format                 # Format code with black
make format-check           # Check if code is properly formatted
make lint                   # Run linting with flake8
make check                  # Run format-check, lint, and test

# Environment Management
make install                # Create .venv and install dependencies
make dev-install            # Install in development mode
make clean                  # Clean temporary files
make clean-all              # Clean everything including .venv
make reinstall              # Clean and reinstall everything

# Development Workflow
make dev                    # Run format, lint, and test
make status                 # Show project status
make env-info               # Show environment information

# Package Management
make build                  # Build the package
make publish-test           # Publish to Test PyPI
make publish                # Publish to PyPI
```

### Manual Setup (without Makefile)

If you prefer not to use the Makefile:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=llm_libration --cov-report=html --cov-report=term

# Format code
black llm_libration tests

# Check formatting
black --check llm_libration tests

# Run linter
flake8 llm_libration tests
```

### Project Structure

```
llm-libration/
├── llm_libration/          # Main package
│   ├── __init__.py         # Package initialization
│   ├── analyzer.py         # Main analyzer class
│   ├── types.py           # Type definitions and enums
│   └── exceptions.py      # Custom exceptions
├── tests/                 # Test suite
│   ├── __init__.py
│   └── test_analyzer.py   # Main test file
├── .venv/                 # Virtual environment (hidden)
├── Makefile              # Development commands
├── pyproject.toml         # Package configuration
├── .env.dist             # Environment template
└── README.md             # This file
```

## Requirements

-   Python 3.8+
-   OpenAI API key
-   Dependencies: langchain, python-dotenv, pillow

## License

MIT License - see the LICENSE file for details.

## Author

Evgeny Smirnov

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `make check`
6. Submit a pull request

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**: Make sure you've created a `.env` file with your API key
2. **Image loading errors**: Ensure your image files are valid and readable
3. **LLM response errors**: The model might return unexpected responses for unusual images

### Debug Mode

For debugging, you can increase the temperature and examine raw LLM responses:

```python
analyzer = ResonanceAnalyzer(temperature=0.1)
# Add logging to see raw responses
```

### Getting Help

```bash
# Check project status
make status

# Show environment information
make env-info

# Show all available commands
make help
```
