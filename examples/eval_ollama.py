"""
Backward-compatible wrapper for the unified evaluator.

DEPRECATED: Use unified_eval.py instead
    python -m examples.unified_eval    # Uses Ollama by default

This script is maintained for backward compatibility.
It delegates to the unified unified_eval.py with LLM_PROVIDER=ollama.
"""

from ._eval_wrapper import run_evaluator

if __name__ == "__main__":
    run_evaluator("ollama")