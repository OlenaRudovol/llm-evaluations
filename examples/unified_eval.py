"""
Unified LLM evaluator that works with any provider (Ollama, OpenAI, etc.).

The provider and model are configured via environment variables:
- LLM_PROVIDER: 'ollama' (default) or 'openai'
- OLLAMA_MODEL: Model for Ollama (default: gemma3:1b)
- OPENAI_MODEL: Model for OpenAI (default: gpt-4o-mini)
- OPENAI_API_KEY: Required for OpenAI provider
- OLLAMA_HOST: Optional Ollama host (default: localhost)
- OLLAMA_PORT: Optional Ollama port (default: 11434)

Usage:
    python -m examples.unified_eval              # Uses default provider (Ollama)
    LLM_PROVIDER=openai python -m examples.unified_eval  # Uses OpenAI
"""

import logging
from pathlib import Path

from src.llm_eval.core import evaluator, config
from src.llm_eval.data import DataLoader
from src.llm_eval.adapters import OllamaAdapter, OpenAIAdapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load test data from external file
DATA_FILE = Path(__file__).parent.parent / "data" / "car_color_samples.jsonl"


def create_adapter():
    """Create the appropriate LLM adapter based on configuration."""
    provider = config.llm_provider.lower()
    
    if provider == "ollama":
        logger.info(f"Using Ollama provider with model '{config.ollama_model}'")
        return OllamaAdapter(model=config.ollama_model)
    elif provider == "openai":
        logger.info(f"Using OpenAI provider with model '{config.openai_model}'")
        return OpenAIAdapter(
            model=config.openai_model,
            api_key=config.openai_api_key,
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider}. "
            f"Valid options are: 'ollama', 'openai'"
        )


def evaluate():
    """Run evaluation with the configured LLM provider."""
    adapter = create_adapter()
    
    logger.info(f"Loading test data from {DATA_FILE}")
    
    # Stream data instead of loading all into memory
    test_cases = [
        (evaluator.generate_question(item), item["expected"])
        for item in DataLoader.stream_jsonl(str(DATA_FILE))
    ]
    
    logger.info(f"Loaded {len(test_cases)} test cases")
    evaluator.simple_exact_match_eval(test_cases, adapter.ask, model_name=adapter.model_name)


if __name__ == "__main__":
    evaluate()