# PEP 8 Style Guide Summary

## Naming Conventions
- `snake_case` for functions, methods, variables, modules
- `PascalCase` for class names
- `UPPER_SNAKE_CASE` for constants
- `_single_leading_underscore` for internal use
- `__double_leading_underscore` for name mangling

## Formatting
- 4 spaces per indentation level (no tabs)
- Maximum line length: 88 characters (Black default) or 79 (strict PEP 8)
- 2 blank lines before top-level definitions
- 1 blank line between method definitions

## Imports
```python
# Standard library
import os
import sys

# Third-party
import torch
import numpy as np

# Local
from chatterbox.tts import ChatterboxTTS
```

- One import per line for `import x`
- Group imports: standard library, third-party, local
- Absolute imports preferred over relative

## Whitespace
```python
# Good
spam(ham[1], {eggs: 2})
x = 1
y = 2

# Bad
spam( ham[ 1 ], { eggs: 2 } )
x=1
y             = 2
```

## Comments
- Use complete sentences
- Block comments: `# ` prefix, same indentation as code
- Inline comments: sparingly, separated by 2+ spaces

## Reference
- Full guide: https://peps.python.org/pep-0008/
