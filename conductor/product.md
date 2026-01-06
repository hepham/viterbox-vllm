# Product Guide

## Overview
Chatterbox TTS is a production-grade open-source text-to-speech library designed for developers who need high-quality, multilingual voice synthesis.

## Vision
To provide the most accessible and capable open-source TTS solution that rivals closed-source commercial offerings, enabling creators to bring content to life across languages with optimized inference performance.

## Target Users
- **Video/Content Creators** - Producing voiceovers, narration, audiobooks, and multimedia content

## Key Features
- **Multilingual Support** - 23 languages out of the box
- **Zero-Shot Voice Cloning** - Clone any voice from a short audio reference without fine-tuning
- **Emotion/Exaggeration Control** - Unique intensity control to adjust expressiveness and dramatic delivery
- **vLLM Integration** - High-performance inference backend with ~4x speedup (10x+ with batching), efficient GPU memory utilization, and state-of-the-art inference infrastructure

## Deployment Model
Python pip package (`chatterbox-tts`) for local or server-side installation. Supports CUDA, MPS (Apple Silicon), and CPU backends. Optional vLLM backend for production-scale throughput.

## Performance Philosophy
Balanced approach prioritizing good output quality with optimized inference speed. vLLM integration removes T3 model as bottleneck, enabling benchmark-topping throughput for batch processing and real-time applications.

## Technical Foundation
- 0.5B parameter Llama backbone
- Alignment-informed inference for ultra-stable output
- Trained on 0.5M hours of cleaned data
- vLLM port for efficient GPU memory and batched generation
- MIT licensed
