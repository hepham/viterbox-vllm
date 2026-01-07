# Migration Guide: Chatterbox to Chatterbox-vLLM

This guide helps you migrate from the original [Chatterbox](https://github.com/resemble-ai/chatterbox) implementation to the vLLM-accelerated version.

## Key Differences

| Feature | Original Chatterbox | Chatterbox-vLLM |
|---------|---------------------|-----------------|
| Backend | HuggingFace Transformers | vLLM |
| Batching | Limited | Full support |
| Speed | Baseline | ~4x faster (10x+ with batching) |
| Platform | Linux, macOS, Windows | Linux/WSL2 only |
| GPU | CUDA, MPS, CPU | NVIDIA CUDA only |

## Installation

**Original:**
```bash
pip install chatterbox-tts
```

**vLLM:**
```bash
git clone https://github.com/randombk/chatterbox-vllm.git
cd chatterbox-vllm
uv venv && source .venv/bin/activate && uv sync
```

## Basic Usage

### Original Chatterbox

```python
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
wav = model.generate("Hello world!")
```

### Chatterbox-vLLM

```python
from chatterbox_vllm.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(
    max_model_len=1000,
    compile=False,  # Set True for CUDA graphs
)
wavs = model.generate(["Hello world!"])  # Always returns list
wav = wavs[0]
```

## Key API Changes

### 1. Return Type

**Original:** `generate()` returns a single tensor
**vLLM:** `generate()` always returns a list of tensors (for batching)

```python
# Original
wav = model.generate("Hello")

# vLLM
wavs = model.generate(["Hello"])
wav = wavs[0]  # Extract single result
```

### 2. Batching

**Original:** Process one prompt at a time
**vLLM:** Native batch processing

```python
# vLLM - process multiple prompts efficiently
prompts = ["Hello!", "How are you?", "Goodbye!"]
wavs = model.generate(prompts)  # All processed together
```

### 3. Model Loading Parameters

**Original:**
```python
model = ChatterboxTTS.from_pretrained(device="cuda")
```

**vLLM:**
```python
model = ChatterboxTTS.from_pretrained(
    max_model_len=1000,      # Max tokens per request
    max_batch_size=10,       # Max concurrent requests
    compile=False,           # CUDA graphs (True for speed)
    s3gen_use_fp16=False,    # FP16 for S3Gen
)
```

### 4. CFG Scale

**Original:** Environment variable only
```bash
export CHATTERBOX_CFG_SCALE=0.5
```

**vLLM:** Per-request or environment variable
```python
wavs = model.generate(prompts, cfg_scale=0.5)
```

### 5. Diffusion Steps

**Original:** Fixed at 10
**vLLM:** Configurable per-request

```python
# Fast generation (lower quality)
wavs = model.generate(prompts, diffusion_steps=5)

# High quality (slower)
wavs = model.generate(prompts, diffusion_steps=10)
```

## Feature Mapping

| Original Feature | vLLM Equivalent |
|------------------|-----------------|
| `model.generate(text)` | `model.generate([text])[0]` |
| `exaggeration=0.5` | `exaggeration=0.5` (same) |
| `cfg_scale` env var | `cfg_scale` param or env var |
| Voice cloning | `audio_prompt_path` (same) |
| Multilingual | `from_pretrained_multilingual()` |

## Not Yet Supported in vLLM

- macOS (MPS) support
- CPU-only inference
- Server/API mode
- Learned speech positional embeddings (minimal quality impact)

## Performance Tips

1. **Use batching**: Process multiple prompts together
2. **Enable CUDA graphs**: Set `compile=True` for repeated inference
3. **Lower diffusion steps**: Use 5 instead of 10 for 2x S3Gen speed
4. **Pre-compute conditioning**: Use `generate_with_conds()` for same voice

## Troubleshooting

### "Module not found" errors
Make sure you're using the vLLM virtual environment:
```bash
source .venv/bin/activate
```

### CUDA out of memory
Reduce batch size or max_model_len:
```python
model = ChatterboxTTS.from_pretrained(
    max_model_len=500,
    max_batch_size=2,
)
```

### Slow first generation
This is CUDA graph compilation. Use `compile=False` for one-off generation.
