#!/usr/bin/env python3
"""
Chatterbox TTS on vLLM - High-performance text-to-speech with vLLM acceleration.

Usage:
    from chatterbox_vllm.tts import ChatterboxTTS
    
    model = ChatterboxTTS.from_pretrained()
    audios = model.generate(["Hello world!"])
"""

from chatterbox_vllm.tts import ChatterboxTTS
from chatterbox_vllm.exceptions import (
    ChatterboxError,
    ModelLoadError,
    InferenceError,
    AudioProcessingError,
    ConfigurationError,
    VLLMError,
)

__version__ = "0.2.0"
__all__ = [
    "ChatterboxTTS",
    "ChatterboxError",
    "ModelLoadError", 
    "InferenceError",
    "AudioProcessingError",
    "ConfigurationError",
    "VLLMError",
]
