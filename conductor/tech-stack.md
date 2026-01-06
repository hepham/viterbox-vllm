# Tech Stack

## Language
- **Python 3.10+** - Primary development language

## Deep Learning
- **PyTorch** - Core tensor operations and neural network framework
- **Torchaudio** - Audio I/O and transformations

## ML Frameworks
- **Transformers** - Hugging Face model loading and inference
- **Diffusers** - Diffusion model components (S3Gen)
- **Conformer** - Conformer architecture blocks

## Inference Optimization
- **vLLM** - Optional high-performance inference backend
  - ~4x speedup without batching, 10x+ with batching
  - Efficient GPU memory utilization
  - Requires Linux/WSL2 with NVIDIA hardware

## Audio Processing
- **librosa** - Audio analysis and feature extraction
- **resampy** - High-quality audio resampling

## Tokenization
- **S3Tokenizer** - Speech token encoding
- **spacy-pkuseg** - Chinese text segmentation
- **pykakasi** - Japanese text romanization

## Serialization & Config
- **safetensors** - Safe and fast model weight serialization
- **omegaconf** - Configuration management

## Watermarking
- **resemble-perth** - Imperceptible neural audio watermarking

## Package Management
- **setuptools** - Standard Python packaging (main package)
- **uv** - Fast Python package installer (vLLM variant)

## Build System
- **pyproject.toml** - PEP 517/518 compliant build configuration
- **src layout** - Source code in `src/chatterbox/`
