"""
Wrapper factory for backward compatibility with separate eval files.

Consolidates eval_ollama.py and eval_openai.py into a single factory function.
"""

import os
from . import eval as evaluator_module


def run_evaluator(provider: str):
    """Run evaluator with specified provider.
    
    Args:
        provider: 'ollama' or 'openai'
    """
    os.environ["LLM_PROVIDER"] = provider
    evaluator_module.evaluate()
