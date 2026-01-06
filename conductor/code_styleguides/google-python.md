# Google Python Style Guide Summary

## Docstrings
Use Google-style docstrings:

```python
def generate(
    self,
    text: str,
    audio_prompt_path: str | None = None,
    exaggeration: float = 0.5,
) -> torch.Tensor:
    """Generate speech audio from text.

    Args:
        text: The text to synthesize.
        audio_prompt_path: Optional path to reference audio for voice cloning.
        exaggeration: Emotion intensity control, 0.0 to 1.0.

    Returns:
        Audio tensor of shape (1, num_samples) at model sample rate.

    Raises:
        FileNotFoundError: If audio_prompt_path doesn't exist.
        ValueError: If exaggeration is outside valid range.
    """
```

## Type Hints
- Use type hints for all public functions
- Use `|` for unions (Python 3.10+): `str | None`
- Use `list[T]`, `dict[K, V]` (lowercase, Python 3.9+)

```python
def process_audio(
    audio: torch.Tensor,
    sample_rate: int = 24000,
) -> tuple[torch.Tensor, int]:
    ...
```

## Error Handling
```python
# Good - specific exceptions with context
if not path.exists():
    raise FileNotFoundError(f"Audio file not found: {path}")

# Bad - bare except
try:
    process()
except:
    pass
```

## Boolean Expressions
```python
# Good
if not users:
    print("No users")

if foo is None:
    handle_none()

# Bad
if len(users) == 0:
    print("No users")

if foo == None:
    handle_none()
```

## Reference
- Full guide: https://google.github.io/styleguide/pyguide.html
