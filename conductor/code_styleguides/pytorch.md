# PyTorch Best Practices

## Tensor Operations

### Device Management
```python
# Good - explicit device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = tensor.to(device)

# Good - use model's device
tensor = tensor.to(next(model.parameters()).device)
```

### Memory Efficiency
```python
# Use inference mode for generation
with torch.inference_mode():
    output = model.generate(input)

# Clear cache when needed
torch.cuda.empty_cache()

# Avoid unnecessary tensor copies
output = model(input)  # Good
output = model(input.clone())  # Bad - unnecessary copy
```

### In-place Operations
```python
# Use in-place when safe (suffix with _)
tensor.add_(1)  # In-place
tensor = tensor + 1  # Creates new tensor
```

## Model Code

### Module Structure
```python
class TTSModel(nn.Module):
    def __init__(self, config: TTSConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.decoder(encoded)
```

### Weight Loading
```python
# Use safetensors for safety and speed
from safetensors.torch import load_file

weights = load_file("model.safetensors")
model.load_state_dict(weights)
```

## Data Types
```python
# Use appropriate dtypes
audio = audio.to(torch.float32)  # Audio processing
weights = weights.to(torch.float16)  # Model weights (if supported)
```

## Batch Processing
```python
# Process in batches for efficiency
batch_size = 32
for i in range(0, len(inputs), batch_size):
    batch = inputs[i:i + batch_size]
    outputs.extend(model(batch))
```

## Reference
- PyTorch docs: https://pytorch.org/docs/stable/
- PyTorch tutorials: https://pytorch.org/tutorials/
