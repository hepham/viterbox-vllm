"""Custom exceptions for Chatterbox vLLM with actionable error messages."""


class ChatterboxError(Exception):
    """Base exception for Chatterbox vLLM errors."""
    pass


class ModelLoadError(ChatterboxError):
    """Error loading the model or its components."""
    
    def __init__(self, component: str, message: str, suggestion: str = None):
        self.component = component
        self.suggestion = suggestion
        full_message = f"Failed to load {component}: {message}"
        if suggestion:
            full_message += f"\n  Suggestion: {suggestion}"
        super().__init__(full_message)


class InferenceError(ChatterboxError):
    """Error during inference."""
    
    def __init__(self, stage: str, message: str, suggestion: str = None):
        self.stage = stage
        self.suggestion = suggestion
        full_message = f"Inference error at {stage}: {message}"
        if suggestion:
            full_message += f"\n  Suggestion: {suggestion}"
        super().__init__(full_message)


class AudioProcessingError(ChatterboxError):
    """Error processing audio input or output."""
    
    def __init__(self, message: str, suggestion: str = None):
        self.suggestion = suggestion
        full_message = f"Audio processing error: {message}"
        if suggestion:
            full_message += f"\n  Suggestion: {suggestion}"
        super().__init__(full_message)


class ConfigurationError(ChatterboxError):
    """Invalid configuration or parameters."""
    
    def __init__(self, param: str, value, message: str, valid_range: str = None):
        self.param = param
        self.value = value
        self.valid_range = valid_range
        full_message = f"Invalid configuration for '{param}' = {value}: {message}"
        if valid_range:
            full_message += f"\n  Valid range: {valid_range}"
        super().__init__(full_message)


class VLLMError(ChatterboxError):
    """Error from vLLM backend."""
    
    def __init__(self, message: str, suggestion: str = None):
        self.suggestion = suggestion
        full_message = f"vLLM error: {message}"
        if suggestion:
            full_message += f"\n  Suggestion: {suggestion}"
        super().__init__(full_message)
