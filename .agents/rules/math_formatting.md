# Mathematical Formatting Rule

## Strict Rule on Equations & Math:
1. **Never use raw LaTeX delimiters (`$ ... $`, `$$ ... $$`, `\[ ... \]`, `\( ... \)`)** in chat messages or responses because the chat UI markdown renderer does not support LaTeX parsers.
2. **Always format equations and math using Option A (Unicode Math in formatted code blocks)**:
   - Use standard Unicode mathematical symbols: `α`, `β`, `λ`, `μ`, `θ`, `π`, `∇`, `∑`, `∫`, `∈`, `²`, `³`, `₀`, `₁`, `→`, `·`, `±`, `≈`, `≤`, `≥`.
   - Place standalone equations inside ```text ... ``` or ```python ... ``` code blocks.
   - For inline variables, numbers, or units, use plain text or inline backticks (e.g. `x`, `y`, `theta`, `0.96`, `0.0525 m`).
