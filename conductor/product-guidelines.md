# Product Guidelines

## Brand Voice & Tone
- **Technical and precise** - Developer-focused documentation style
- Prioritize accuracy and clarity over marketing language
- Use concrete examples over abstract descriptions

## Documentation Standards

### Priority
1. **API Reference** - Complete function signatures, parameters, return types, and code examples
2. **Conceptual Explanations** - Architecture overviews, how components interact
3. **Tutorials** - Step-by-step guides for common use cases

### Style
- Lead with working code examples
- Explain the "why" after showing the "how"
- Include expected outputs where applicable

## Error Handling
- **Verbose with actionable suggestions**
- Error messages should explain what went wrong and how to fix it
- Include relevant context (e.g., expected vs actual values)
- Example: `"Audio file not found at '{path}'. Ensure the file exists and the path is absolute."`

## Code Conventions

### Naming (PEP 8)
- `snake_case` for functions, methods, and variables
- `PascalCase` for class names
- `UPPER_SNAKE_CASE` for constants
- Descriptive names over abbreviations

### Examples
```python
# Good
def generate_speech(text: str, audio_prompt_path: str = None) -> torch.Tensor:
    ...

class ChatterboxTTS:
    DEFAULT_SAMPLE_RATE = 24000
```

## Versioning
- **Semantic Versioning** (MAJOR.MINOR.PATCH)
- Breaking changes require MAJOR version bump
- Clear deprecation warnings before removal
- Breaking changes documented prominently in release notes
