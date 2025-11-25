# Design Specification: HuggingFace Transformers Provider with Quantization Support

## 1. Overview

### 1.1 Objective

Add a new `huggingface` (or `hf`) provider to llm-libration that enables running vision-language models locally using HuggingFace Transformers, with optional 4-bit/8-bit quantization via bitsandbytes.

### 1.2 Scope

- New provider type: `huggingface` / `hf`
- Support for vision-language models (VLMs) that can analyze images
- Optional quantization: `none`, `4bit`, `8bit`
- Integration with existing `LLMClient` architecture

## 2. Architecture Design

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      LibrationAnalyzer                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        LLMClient                                │
│  ┌──────────┬──────────┬───────────┬─────────┬───────────────┐ │
│  │  OpenAI  │Anthropic │OpenRouter │ Ollama  │ HuggingFace   │ │
│  │          │          │           │         │ (NEW)         │ │
│  └──────────┴──────────┴───────────┴─────────┴───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                                      │
                          ┌───────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  HuggingFaceVisionClient                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • AutoModelForImageTextToText / Pipeline               │   │
│  │  • AutoProcessor                                         │   │
│  │  • BitsAndBytesConfig (optional quantization)           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Image Path → encode_image() → PIL.Image
                    │
                    ▼
            processor.apply_chat_template()
                    │
                    ▼
            model.generate()
                    │
                    ▼
            processor.decode() → JSON string
                    │
                    ▼
            _clean_json_response() → LibrationAnalysisResult
```

## 3. Configuration Design

### 3.1 Environment Variables

```bash
# Provider selection
LLM_PROVIDER=huggingface

# HuggingFace Configuration
HF_MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct       # Default VLM
HF_QUANTIZATION=4bit                           # none | 4bit | 8bit
HF_DEVICE_MAP=auto                             # auto | cuda:0 | cpu
HF_TORCH_DTYPE=bfloat16                        # float16 | bfloat16 | float32
HF_MAX_NEW_TOKENS=256                          # Generation limit

# 4-bit specific options (when HF_QUANTIZATION=4bit)
HF_4BIT_QUANT_TYPE=nf4                         # nf4 | fp4
HF_4BIT_USE_DOUBLE_QUANT=true                  # Enable nested quantization
HF_4BIT_COMPUTE_DTYPE=bfloat16                 # Compute dtype for 4-bit
```

### 3.2 Config Class Extensions

**File:** `llm_libration/config.py`

```python
# Add to LLMProvider type
LLMProvider = Literal["openai", "anthropic", "openrouter", "ollama", "huggingface"]

class Config:
    # ... existing properties ...

    # HuggingFace Configuration
    @property
    def hf_model_name(self) -> str:
        """Get HuggingFace model name."""
        return os.getenv("HF_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")

    @property
    def hf_quantization(self) -> str:
        """Get quantization mode: none, 4bit, or 8bit."""
        return os.getenv("HF_QUANTIZATION", "none").lower()

    @property
    def hf_device_map(self) -> str:
        """Get device mapping strategy."""
        return os.getenv("HF_DEVICE_MAP", "auto")

    @property
    def hf_torch_dtype(self) -> str:
        """Get torch dtype for model loading."""
        return os.getenv("HF_TORCH_DTYPE", "bfloat16")

    @property
    def hf_max_new_tokens(self) -> int:
        """Get max new tokens for generation."""
        return int(os.getenv("HF_MAX_NEW_TOKENS", "256"))

    @property
    def hf_4bit_quant_type(self) -> str:
        """Get 4-bit quantization type: nf4 or fp4."""
        return os.getenv("HF_4BIT_QUANT_TYPE", "nf4")

    @property
    def hf_4bit_use_double_quant(self) -> bool:
        """Whether to use double quantization for 4-bit."""
        return os.getenv("HF_4BIT_USE_DOUBLE_QUANT", "true").lower() == "true"

    @property
    def hf_4bit_compute_dtype(self) -> str:
        """Get compute dtype for 4-bit quantization."""
        return os.getenv("HF_4BIT_COMPUTE_DTYPE", "bfloat16")
```

## 4. LLMClient Extension

### 4.1 New Imports (Conditional)

**File:** `llm_libration/llm/client.py`

```python
# Add to existing imports
try:
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    AutoModelForImageTextToText = None
    AutoProcessor = None
    BitsAndBytesConfig = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
```

### 4.2 LLMClient.__init__ Extension

```python
def __init__(self, provider: str, model_name: str, api_key: str = "", base_url: str = "",
             quantization: str = "none", device_map: str = "auto", torch_dtype: str = "bfloat16"):
    # ... existing code for other providers ...

    elif self.provider == "huggingface":
        if not HF_AVAILABLE:
            raise ConfigurationError(
                "HuggingFace provider requested but 'transformers' is not installed. "
                "Install with: pip install transformers accelerate"
            )
        if not TORCH_AVAILABLE:
            raise ConfigurationError(
                "HuggingFace provider requires PyTorch. Install with: pip install torch"
            )

        self.quantization = quantization
        self.device_map = device_map
        self.torch_dtype = self._resolve_torch_dtype(torch_dtype)
        self.hf_model, self.hf_processor = self._init_huggingface_model(model_name)
```

### 4.3 HuggingFace Initialization Method

```python
def _resolve_torch_dtype(self, dtype_str: str):
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)

def _init_huggingface_model(self, model_name: str):
    """Initialize HuggingFace vision-language model with optional quantization."""
    logger.info(f"Loading HuggingFace model: {model_name} (quantization={self.quantization})")

    # Build quantization config if needed
    quantization_config = None
    if self.quantization == "4bit":
        if BitsAndBytesConfig is None:
            raise ConfigurationError(
                "4-bit quantization requires bitsandbytes. "
                "Install with: pip install bitsandbytes"
            )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.hf_4bit_quant_type,
            bnb_4bit_use_double_quant=config.hf_4bit_use_double_quant,
            bnb_4bit_compute_dtype=self._resolve_torch_dtype(config.hf_4bit_compute_dtype),
        )
    elif self.quantization == "8bit":
        if BitsAndBytesConfig is None:
            raise ConfigurationError(
                "8-bit quantization requires bitsandbytes. "
                "Install with: pip install bitsandbytes"
            )
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)

    # Load model with configuration
    model_kwargs = {
        "device_map": self.device_map,
        "torch_dtype": self.torch_dtype,
    }
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)

    logger.info(f"Model loaded successfully on device: {model.device}")
    return model, processor
```

### 4.4 Analysis Method for HuggingFace

```python
def _analyze_with_huggingface(self, image_path: Union[str, Path], prompt: str) -> LibrationAnalysisResult:
    """
    Analyze image using HuggingFace vision-language model.

    Args:
        image_path: Path to the image file
        prompt: Analysis prompt

    Returns:
        Structured libration analysis result
    """
    try:
        # Load image
        image = Image.open(image_path)

        # Format messages for chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self._format_prompt_for_json(prompt)},
                ],
            }
        ]

        # Apply chat template and process
        inputs = self.hf_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Add image to inputs
        image_inputs = self.hf_processor(images=image, return_tensors="pt")
        inputs.update(image_inputs)

        # Move to model device
        inputs = {k: v.to(self.hf_model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            generated_ids = self.hf_model.generate(
                **inputs,
                max_new_tokens=config.hf_max_new_tokens,
                do_sample=False,
            )

        # Decode output (skip input tokens)
        input_len = inputs["input_ids"].shape[1]
        output_text = self.hf_processor.decode(
            generated_ids[0, input_len:],
            skip_special_tokens=True
        )

        logger.debug(f"HuggingFace raw output: {output_text}")

        # Parse JSON response
        cleaned_content = self._clean_json_response(output_text)
        parsed_data = json.loads(cleaned_content)

        if 'status' not in parsed_data or 'subtype' not in parsed_data:
            raise ValueError(f"Missing required fields in response: {parsed_data}")

        return LibrationAnalysisResult(**parsed_data)

    except ImageAnalysisError:
        raise
    except Exception as e:
        raise LLMResponseError(f"HuggingFace analysis failed: {str(e)}")
```

### 4.5 Update analyze_image_with_prompt

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
        provider: LLM provider (openai, anthropic, openrouter, ollama, huggingface)
        quantization: For HuggingFace only: none, 4bit, or 8bit
    """
    config.validate_required_env_vars()

    provider = provider or config.llm_provider

    # ... existing provider setup ...

    elif provider == "huggingface":
        model_name = model_name or config.hf_model_name
        quantization = quantization or config.hf_quantization

    # ... rest of init ...

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
    type=click.Choice(['openai', 'anthropic', 'openrouter', 'ollama', 'huggingface'], case_sensitive=False),
    help='LLM provider to use (default: openai)',
)
@click.option(
    '--quantization',
    default=None,
    type=click.Choice(['none', '4bit', '8bit'], case_sensitive=False),
    help='Quantization mode for HuggingFace provider (default: from env or none)',
)
def run(image_files, provider, model_name, quantization, ...):
    # Pass quantization to analyzer
    init_kwargs = {'provider': provider}
    if model_name:
        init_kwargs['model_name'] = model_name
    if quantization and provider == 'huggingface':
        init_kwargs['quantization'] = quantization

    analyzer = LibrationAnalyzer(**init_kwargs)
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
dev = [
    "black>=23.0.0",
    "flake8>=6.0.0",
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0"
]
```

### 7.2 Installation Commands

```bash
# Basic HuggingFace support
pip install llm-libration[huggingface]

# With quantization support
pip install llm-libration[huggingface-quantized]
```

## 8. Recommended VLM Models

| Model | Size | Use Case | Quantization Recommendation |
|-------|------|----------|----------------------------|
| `Qwen/Qwen2-VL-2B-Instruct` | 2B | Development, low VRAM | none or 8bit |
| `Qwen/Qwen2-VL-7B-Instruct` | 7B | Production balance | 4bit |
| `OpenGVLab/InternVL3-1B-hf` | 1B | Fast inference | none |
| `OpenGVLab/InternVL3-8B-hf` | 8B | High accuracy | 4bit |
| `llava-hf/llava-v1.6-mistral-7b-hf` | 7B | General purpose | 4bit |

## 9. File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `llm_libration/config.py` | Modify | Add HF config properties |
| `llm_libration/llm/client.py` | Modify | Add HF provider support |
| `llm_libration/analyzer.py` | Modify | Add quantization param |
| `llm_libration/cli/run.py` | Modify | Add --quantization flag |
| `llm_libration/cli/benchmark.py` | Modify | Add --quantization flag |
| `pyproject.toml` | Modify | Add optional deps |
| `.env.dist` | Modify | Add HF env vars |
| `tests/test_llm_client.py` | Modify | Add HF provider tests |

## 10. Usage Examples

### 10.1 CLI Usage

```bash
# With default model (no quantization)
llm-libration run image.png --provider huggingface

# With 4-bit quantization
llm-libration run image.png --provider huggingface --quantization 4bit

# With custom model
llm-libration run image.png --provider huggingface \
    --model Qwen/Qwen2-VL-2B-Instruct --quantization 8bit

# Benchmark with quantized model
llm-libration benchmark ./benchmark_data --provider huggingface --quantization 4bit
```

### 10.2 Python API

```python
from llm_libration import LibrationAnalyzer

# Default (no quantization)
analyzer = LibrationAnalyzer(provider="huggingface")

# With 4-bit quantization
analyzer = LibrationAnalyzer(
    provider="huggingface",
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    quantization="4bit"
)

# Analyze
result = analyzer.analyze_image("resonance_plot.png")
print(f"Status: {result.status}, Subtype: {result.subtype}")
```

## 11. Memory Requirements

| Configuration | ~VRAM Required |
|--------------|----------------|
| 7B model, no quant | 14-16 GB |
| 7B model, 8-bit | 8-10 GB |
| 7B model, 4-bit | 5-6 GB |
| 2B model, no quant | 4-5 GB |
| 2B model, 4-bit | 2-3 GB |

## 12. Validation Checklist

- [ ] HuggingFace provider initializes correctly without bitsandbytes (no quantization)
- [ ] 4-bit quantization works with bitsandbytes installed
- [ ] 8-bit quantization works with bitsandbytes installed
- [ ] Graceful error when dependencies missing
- [ ] JSON parsing handles VLM output formats
- [ ] CLI --quantization flag works correctly
- [ ] Environment variables are respected
- [ ] Memory usage within expected bounds
