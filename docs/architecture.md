# Chatterbox-vLLM Architecture

## Overview

Chatterbox-vLLM is a high-performance TTS system that combines vLLM for efficient LLM inference with PyTorch-based audio generation.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Input Layer                                │
├──────────────────────────────┬──────────────────────────────────────┤
│      Text Prompts            │      Voice Reference Audio           │
└──────────────┬───────────────┴──────────────────┬───────────────────┘
               │                                   │
               ▼                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Preprocessing                                  │
├─────────────────────┬────────────────────┬──────────────────────────┤
│   Text Tokenizer    │   Voice Encoder    │    S3 Tokenizer          │
│ (En/MTL/Viterbox)   │    (CAMPPlus)      │  (Speech Tokens)         │
└─────────┬───────────┴─────────┬──────────┴─────────────┬────────────┘
          │                     │                         │
          └──────────┬──────────┴─────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    T3 Model (vLLM Backend)                           │
├──────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐ │
│  │  T3CondEnc      │ -> │  LLaMA 520M      │ -> │  CFG Processing │ │
│  │  (Conditioning) │    │  (Token Gen)     │    │  (Cond+Uncond)  │ │
│  └─────────────────┘    └──────────────────┘    └────────┬────────┘ │
│                                                          │          │
│                                    ┌─────────────────────▼────────┐ │
│                                    │  Alignment Analyzer          │ │
│                                    │  (Repetition/Hallucination)  │ │
│                                    └─────────────────────┬────────┘ │
└──────────────────────────────────────────────────────────┼──────────┘
                                                           │
                                                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    S3Gen (PyTorch Backend)                           │
├──────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────┐    ┌───────────────────────────┐   │
│  │  Flow Matching (CFM)        │ -> │  HiFiGAN Vocoder          │   │
│  │  (Speech Tokens → Mel)      │    │  (Mel → Waveform)         │   │
│  └─────────────────────────────┘    └──────────────┬────────────┘   │
└─────────────────────────────────────────────────────┼───────────────┘
                                                      │
                                                      ▼
                                            ┌─────────────────┐
                                            │  Audio Output   │
                                            │  (24kHz WAV)    │
                                            └─────────────────┘
```

## Component Details

### Text Tokenizers

| Tokenizer | Use Case | Vocab Size |
|-----------|----------|------------|
| EnTokenizer | English only | 704 |
| MTLTokenizer | Multilingual (23 languages) | 2454 |
| ViterboxTokenizer | Vietnamese + English | 2549 |

### T3 Model (vLLM)

The T3 model is a 520M parameter LLaMA-based model that generates speech tokens from text. Key features:

- **Conditioning Encoder (T3CondEnc)**: Encodes voice reference, text, and exaggeration into embeddings
- **CFG Processing**: Runs conditioned and unconditioned inference for classifier-free guidance
- **Alignment Analyzer**: Detects and prevents repetition/hallucination issues

### S3Gen (PyTorch)

Converts speech tokens to audio waveforms:

1. **Flow Matching (CFM)**: Transforms speech tokens to mel-spectrograms using conditional flow matching
2. **HiFiGAN**: Neural vocoder that synthesizes waveforms from mel-spectrograms

### Performance Bottleneck

With vLLM acceleration, **S3Gen is now the bottleneck** (not T3):

- T3 (vLLM): ~15% of generation time
- S3Gen (PyTorch): ~70% of generation time
- Preprocessing: ~15% of generation time

## Data Flow

1. **Text** → Tokenized to text tokens
2. **Voice Reference** → Encoded to speaker embedding + speech token reference
3. **Conditioning** → Combined into T3 input embeddings
4. **T3 Generation** → Produces speech tokens (with CFG)
5. **S3Gen** → Converts speech tokens to mel-spectrogram
6. **HiFiGAN** → Synthesizes final waveform

## Key Files

| Component | File Path |
|-----------|-----------|
| Main TTS Class | `src/chatterbox_vllm/tts.py` |
| T3 vLLM Model | `src/chatterbox_vllm/models/t3/t3.py` |
| Alignment Analyzer | `src/chatterbox_vllm/models/t3/inference/alignment_stream_analyzer_vllm.py` |
| S3Gen | `src/chatterbox_vllm/models/s3gen/s3gen.py` |
| Tokenizers | `src/chatterbox_vllm/models/t3/{en,mtl,viterbox}tokenizer.py` |
