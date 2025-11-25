# Design Specification: MLX Provider for macOS Apple Silicon

## 1. Overview

### 1.1 Objective

Add a new `mlx` provider to llm-libration that enables running vision-language models natively on macOS Apple Silicon using MLX-VLM, with support for pre-quantized models (4-bit/8-bit).

### 1.2 Key Differences from HuggingFace Provider

| Aspect | HuggingFace | MLX |
|--------|-------------|-----|
| **Platform** | Cross-platform (CUDA/CPU) | macOS Apple Silicon only |
| **Quantization** | Runtime via bitsandbytes | Pre-quantized models from mlx-community |
| **Memory** | Unified GPU memory | Unified Apple Silicon memory |
| **Library** | transformers + bitsandbytes | mlx-vlm |

### 1.3 Scope

- New provider type: `mlx`
- Support for MLX-VLM vision-language models
- Pre-quantized model support (4-bit, 8-bit, bf16)
- macOS-only with graceful fallback on other platforms

## 2. Architecture Design

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                       LibrationAnalyzer                             │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          LLMClient                                  │
│  ┌────────┬──────────┬───────────┬────────┬────────────┬─────────┐ │
│  │ OpenAI │Anthropic │OpenRouter │ Ollama │HuggingFace │   MLX   │ │
│  │        │          │           │        │            │  (NEW)  │ │
│  └────────┴──────────┴───────────┴────────┴────────────┴─────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                                              │
                              ┌────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MLX Vision Client                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • mlx_vlm.load() - Model + Processor loading               │   │
│  │  • mlx_vlm.generate() - Image analysis                      │   │
│  │  • apply_chat_template() - Prompt formatting                │   │
│  │  • Pre-quantized models (4bit/8bit/bf16)                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Image Path → PIL.Image.open()
                  │
                  ▼
         apply_chat_template(processor, config, prompt, num_images)
                  │
                  ▼
         generate(model, processor, formatted_prompt, [image])
                  │
                  ▼
         Raw text output → _clean_json_response()
                  │
                  ▼
         LibrationAnalysisResult
```

## 3. Configuration Design

### 3.1 Environment Variables

```bash
# Provider selection
LLM_PROVIDER=mlx

# MLX Configuration
MLX_MODEL_NAME=mlx-community/Qwen2-VL-2B-Instruct-4bit   # Default model
MLX_MAX_TOKENS=256                                        # Generation limit
MLX_TEMPERATURE=0.0                                       # Sampling temperature
MLX_VERBOSE=false                                         # Enable verbose output
```

### 3.2 Config Class Extensions

**File:** `llm_libration/config.py`

```python
# Update LLMProvider type
LLMProvider = Literal["openai", "anthropic", "openrouter", "ollama", "huggingface", "mlx"]

class Config:
    # ... existing properties ...

    # MLX Configuration
    @property
    def mlx_model_name(self) -> str:
        """Get MLX model name (from mlx-community)."""
        return os.getenv("MLX_MODEL_NAME", "mlx-community/Qwen2-VL-2B-Instruct-4bit")

    @property
    def mlx_max_tokens(self) -> int:
        """Get max new tokens for MLX generation."""
        return int(os.getenv("MLX_MAX_TOKENS", "256"))

    @property
    def mlx_temperature(self) -> float:
        """Get temperature for MLX generation."""
        return float(os.getenv("MLX_TEMPERATURE", "0.0"))

    @property
    def mlx_verbose(self) -> bool:
        """Whether to enable verbose MLX output."""
        return os.getenv("MLX_VERBOSE", "false").lower() == "true"
```

### 3.3 Validation Extension

```python
def validate_required_env_vars(self):
    """Validate required environment variables based on provider."""
    provider = self.llm_provider

    # ... existing validations ...

    elif provider == "mlx":
        # MLX doesn't require API keys, but check platform
        import platform
        if platform.system() != "Darwin":
            raise ConfigurationError(
                "MLX provider is only available on macOS. "
                "Use 'huggingface' or 'ollama' provider on other platforms."
            )
        # Check for Apple Silicon
        if platform.machine() not in ("arm64", "aarch64"):
            raise ConfigurationError(
                "MLX provider requires Apple Silicon (M1/M2/M3/M4). "
                "Use 'huggingface' or 'ollama' provider on Intel Macs."
            )
```

## 4. LLMClient Extension

### 4.1 New Imports (Conditional)

**File:** `llm_libration/llm/client.py`

```python
# Add to existing imports
try:
    from mlx_vlm import load as mlx_load, generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template as mlx_apply_chat_template
    from mlx_vlm.utils import load_config as mlx_load_config
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mlx_load = None
    mlx_generate = None
    mlx_apply_chat_template = None
    mlx_load_config = None
```

### 4.2 LLMClient.__init__ Extension

```python
def __init__(self, provider: str, model_name: str, api_key: str = "", base_url: str = "",
             quantization: str = "none", device_map: str = "auto", torch_dtype: str = "bfloat16"):
    """
    Initialize the LLM client.

    Args:
        provider: LLM provider (openai, anthropic, openrouter, ollama, huggingface, mlx)
        model_name: Name of the model to use
        api_key: API key for the provider (not needed for Ollama/MLX)
        base_url: Base URL for the provider
        quantization: Quantization mode for HuggingFace (none, 4bit, 8bit)
        device_map: Device mapping strategy for HuggingFace
        torch_dtype: Torch dtype for HuggingFace
    """
    self.provider = provider.lower()
    self.model_name = model_name

    # ... existing provider initialization ...

    elif self.provider == "mlx":
        if not MLX_AVAILABLE:
            raise ConfigurationError(
                "MLX provider requested but 'mlx-vlm' is not installed. "
                "Install with: pip install mlx-vlm"
            )
        self._validate_mlx_platform()
        self.mlx_model, self.mlx_processor = self._init_mlx_model(model_name)
        self.mlx_config = mlx_load_config(model_name)
```

### 4.3 Platform Validation

```python
@staticmethod
def _validate_mlx_platform():
    """Validate that we're running on Apple Silicon macOS."""
    import platform

    if platform.system() != "Darwin":
        raise ConfigurationError(
            "MLX provider is only available on macOS. "
            "Current platform: " + platform.system()
        )

    if platform.machine() not in ("arm64", "aarch64"):
        raise ConfigurationError(
            "MLX provider requires Apple Silicon (M1/M2/M3/M4). "
            "Current architecture: " + platform.machine()
        )
```

### 4.4 MLX Initialization Method

```python
def _init_mlx_model(self, model_name: str):
    """
    Initialize MLX vision-language model.

    Args:
        model_name: Model path (e.g., 'mlx-community/Qwen2-VL-2B-Instruct-4bit')

    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading MLX model: {model_name}")

    try:
        model, processor = mlx_load(model_name)
        logger.info(f"MLX model loaded successfully")
        return model, processor
    except Exception as e:
        raise ConfigurationError(f"Failed to load MLX model '{model_name}': {str(e)}")
```

### 4.5 Analysis Method for MLX

```python
def _analyze_with_mlx(self, image_path: Union[str, Path], prompt: str) -> LibrationAnalysisResult:
    """
    Analyze image using MLX vision-language model.

    Args:
        image_path: Path to the image file
        prompt: Analysis prompt

    Returns:
        Structured libration analysis result
    """
    try:
        # Validate image exists
        image_path = Path(image_path)
        if not image_path.exists():
            raise ImageAnalysisError(f"Image file not found: {image_path}")

        # Format prompt for JSON output
        formatted_prompt = self._format_prompt_for_json(prompt)

        # Apply chat template
        chat_prompt = mlx_apply_chat_template(
            self.mlx_processor,
            self.mlx_config,
            formatted_prompt,
            num_images=1
        )

        # Generate response
        # MLX-VLM accepts image paths directly as strings
        output = mlx_generate(
            self.mlx_model,
            self.mlx_processor,
            chat_prompt,
            [str(image_path)],
            max_tokens=config.mlx_max_tokens,
            temp=config.mlx_temperature,
            verbose=config.mlx_verbose,
        )

        # Extract text from output
        output_text = output if isinstance(output, str) else str(output)
        logger.debug(f"MLX raw output: {output_text}")

        # Parse JSON response
        cleaned_content = self._clean_json_response(output_text)
        parsed_data = json.loads(cleaned_content)

        if 'status' not in parsed_data or 'subtype' not in parsed_data:
            raise ValueError(f"Missing required fields in response: {parsed_data}")

        return LibrationAnalysisResult(**parsed_data)

    except ImageAnalysisError:
        raise
    except Exception as e:
        raise LLMResponseError(f"MLX analysis failed: {str(e)}")
```

### 4.6 Update analyze_image_with_prompt

```python
def analyze_image_with_prompt(self, image_path: Union[str, Path], prompt: str) -> LibrationAnalysisResult:
    """Analyze an image using the provided prompt."""
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
```

## 5. Analyzer Integration

### 5.1 LibrationAnalyzer.__init__ Extension

**File:** `llm_libration/analyzer.py`

```python
def __init__(self, model_name: str = None, provider: str = None,
             quantization: str = None):
    """
    Initialize the LibrationAnalyzer.

    Args:
        model_name: Name of the model to use
        provider: LLM provider (openai, anthropic, openrouter, ollama, huggingface, mlx)
        quantization: For HuggingFace only: none, 4bit, or 8bit
    """
    config.validate_required_env_vars()

    provider = provider or config.llm_provider

    api_key = ""
    base_url = ""

    if provider == "openai":
        api_key, model_name = config.openai_api_key, model_name or config.openai_model_name
    elif provider == "anthropic":
        api_key, model_name = config.anthropic_api_key, model_name or config.anthropic_model_name
    elif provider == "openrouter":
        api_key, model_name, base_url = (
            config.openrouter_api_key,
            model_name or config.openrouter_model_name,
            config.openrouter_base_url,
        )
    elif provider == "ollama":
        model_name, base_url = model_name or config.ollama_model_name, config.ollama_base_url
    elif provider == "huggingface":
        model_name = model_name or config.hf_model_name
        quantization = quantization or config.hf_quantization
    elif provider == "mlx":
        model_name = model_name or config.mlx_model_name
    else:
        raise ConfigurationError(f"Unsupported provider: {provider}")

    self.llm_client = LLMClient(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        quantization=quantization if provider == "huggingface" else "none",
        device_map=config.hf_device_map if provider == "huggingface" else "auto",
        torch_dtype=config.hf_torch_dtype if provider == "huggingface" else "bfloat16",
    )
```

## 6. CLI Integration

### 6.1 Updated CLI Options

**File:** `llm_libration/cli/run.py` and `benchmark.py`

```python
@click.option(
    '--provider',
    default='openai',
    type=click.Choice(['openai', 'anthropic', 'openrouter', 'ollama', 'huggingface', 'mlx'], case_sensitive=False),
    help='LLM provider to use (default: openai). Use "mlx" for Apple Silicon Macs.',
)
```

## 7. Dependencies

### 7.1 pyproject.toml Updates

```toml
[project.optional-dependencies]
huggingface = [
    "transformers>=4.45.0",
    "accelerate>=0.26.0",
    "torch>=2.0.0",
]
huggingface-quantized = [
    "transformers>=4.45.0",
    "accelerate>=0.26.0",
    "torch>=2.0.0",
    "bitsandbytes>=0.43.0",
]
mlx = [
    "mlx>=0.12.0",
    "mlx-vlm>=0.1.0",
]
dev = [
    "black>=23.0.0",
    "flake8>=6.0.0",
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]
all-local = [
    "mlx>=0.12.0",
    "mlx-vlm>=0.1.0",
    "transformers>=4.45.0",
    "accelerate>=0.26.0",
    "torch>=2.0.0",
    "bitsandbytes>=0.43.0",
]
```

### 7.2 Installation Commands

```bash
# MLX support (macOS Apple Silicon only)
pip install llm-libration[mlx]

# HuggingFace support (cross-platform)
pip install llm-libration[huggingface]

# HuggingFace with quantization
pip install llm-libration[huggingface-quantized]

# All local providers
pip install llm-libration[all-local]
```

## 8. Recommended MLX Models

| Model | Size | Quantization | VRAM | Use Case |
|-------|------|--------------|------|----------|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | 2B | 4-bit | ~2 GB | Development, fast inference |
| `mlx-community/Qwen2-VL-2B-Instruct-8bit` | 2B | 8-bit | ~3 GB | Better accuracy |
| `mlx-community/Qwen2-VL-7B-Instruct-4bit` | 7B | 4-bit | ~5 GB | Production balance |
| `mlx-community/Qwen2-VL-7B-Instruct-8bit` | 7B | 8-bit | ~9 GB | High accuracy |
| `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` | 7B | 4-bit | ~5 GB | Latest Qwen VL |
| `mlx-community/pixtral-12b-4bit` | 12B | 4-bit | ~8 GB | High capability |
| `mlx-community/Molmo-7B-D-0924-4bit` | 7B | 4-bit | ~5 GB | Object detection |
| `mlx-community/llava-interleave-qwen-0.5b-bf16` | 0.5B | bf16 | ~1 GB | Ultra-fast, low memory |

## 9. .env.dist Updates

```bash
# ===========================================
# MLX Configuration (macOS Apple Silicon)
# ===========================================
# Use 'mlx' provider for native Apple Silicon inference
# LLM_PROVIDER=mlx

# MLX model from mlx-community (pre-quantized)
# Available models: https://huggingface.co/mlx-community
MLX_MODEL_NAME=mlx-community/Qwen2-VL-2B-Instruct-4bit

# Generation parameters
MLX_MAX_TOKENS=256
MLX_TEMPERATURE=0.0
MLX_VERBOSE=false
```

## 10. Usage Examples

### 10.1 CLI Usage

```bash
# Basic MLX usage (macOS only)
llm-libration run image.png --provider mlx

# With custom model
llm-libration run image.png --provider mlx \
    --model mlx-community/Qwen2-VL-7B-Instruct-4bit

# Benchmark with MLX
llm-libration benchmark ./benchmark_data --provider mlx

# Compare providers
llm-libration run image.png --provider all  # Now includes mlx on macOS
```

### 10.2 Python API

```python
from llm_libration import LibrationAnalyzer

# Default MLX model
analyzer = LibrationAnalyzer(provider="mlx")

# Custom model
analyzer = LibrationAnalyzer(
    provider="mlx",
    model_name="mlx-community/Qwen2-VL-7B-Instruct-4bit"
)

# Analyze
result = analyzer.analyze_image("resonance_plot.png")
print(f"Status: {result.status}, Subtype: {result.subtype}")
```

### 10.3 Auto-Detection Pattern

```python
import platform

def get_optimal_local_provider():
    """Auto-detect the best local provider for the current platform."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"  # Apple Silicon Mac
    else:
        return "huggingface"  # Other platforms

# Usage
provider = get_optimal_local_provider()
analyzer = LibrationAnalyzer(provider=provider)
```

## 11. Memory Requirements (Apple Silicon)

| Mac Model | Unified Memory | Recommended Model |
|-----------|---------------|-------------------|
| M1/M2 8GB | 8 GB | `Qwen2-VL-2B-Instruct-4bit` |
| M1/M2 16GB | 16 GB | `Qwen2-VL-7B-Instruct-4bit` |
| M1/M2/M3 Pro 18GB+ | 18+ GB | `Qwen2-VL-7B-Instruct-8bit` |
| M1/M2/M3 Max 32GB+ | 32+ GB | `pixtral-12b-4bit` |
| M3 Ultra 64GB+ | 64+ GB | Multiple models / larger |

## 12. File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `llm_libration/config.py` | Modify | Add MLX config properties + platform validation |
| `llm_libration/llm/client.py` | Modify | Add MLX provider support |
| `llm_libration/analyzer.py` | Modify | Handle MLX provider selection |
| `llm_libration/cli/run.py` | Modify | Add 'mlx' to provider choices |
| `llm_libration/cli/benchmark.py` | Modify | Add 'mlx' to provider choices |
| `pyproject.toml` | Modify | Add mlx optional dependency |
| `.env.dist` | Modify | Add MLX env vars |
| `tests/test_llm_client.py` | Modify | Add MLX provider tests |

## 13. Comparison: HuggingFace vs MLX

| Feature | HuggingFace Provider | MLX Provider |
|---------|---------------------|--------------|
| **Platform** | Linux, Windows, macOS | macOS Apple Silicon only |
| **Hardware** | NVIDIA GPU, CPU | Apple M1/M2/M3/M4 |
| **Quantization** | Runtime (bitsandbytes) | Pre-quantized models |
| **Installation** | `pip install llm-libration[huggingface-quantized]` | `pip install llm-libration[mlx]` |
| **Memory Model** | Separate GPU/CPU memory | Unified memory |
| **Typical Startup** | 10-30 seconds | 5-15 seconds |
| **Inference Speed** | Good (GPU dependent) | Excellent (optimized for Apple Silicon) |
| **Model Source** | HuggingFace Hub | mlx-community on HuggingFace |

## 14. Validation Checklist

- [ ] MLX provider initializes correctly on Apple Silicon Mac
- [ ] Graceful error message on non-macOS platforms
- [ ] Graceful error message on Intel Macs
- [ ] Pre-quantized 4-bit models load correctly
- [ ] Pre-quantized 8-bit models load correctly
- [ ] JSON parsing handles MLX output formats
- [ ] CLI provider choice includes 'mlx'
- [ ] Environment variables are respected
- [ ] Memory usage within expected bounds
- [ ] Tests pass with mocked MLX dependencies

## 15. Error Handling

### Platform Errors

```python
# Non-macOS
ConfigurationError: MLX provider is only available on macOS. Current platform: Linux

# Intel Mac
ConfigurationError: MLX provider requires Apple Silicon (M1/M2/M3/M4). Current architecture: x86_64

# Missing dependency
ConfigurationError: MLX provider requested but 'mlx-vlm' is not installed. Install with: pip install mlx-vlm
```

### Model Errors

```python
# Model not found
ConfigurationError: Failed to load MLX model 'invalid-model': [error details]

# Out of memory
LLMResponseError: MLX analysis failed: Out of memory. Try a smaller model or close other applications.
```
