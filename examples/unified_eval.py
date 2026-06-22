"""
Unified LLM evaluator that works with any provider (Ollama, OpenAI, Anthropic).

The provider and model are configured via environment variables:
- LLM_PROVIDER: 'ollama' (default), 'openai', 'anthropic', or 'google'
- OLLAMA_MODEL: Model for Ollama (default: gemma3:1b)
- OPENAI_MODEL: Model for OpenAI (default: gpt-4o-mini)
- OPENAI_API_KEY: Required for OpenAI provider
- ANTHROPIC_MODEL: Model for Anthropic (default: claude-haiku-4-5-20251001)
- ANTHROPIC_API_KEY: Required for Anthropic provider
- GOOGLE_MODEL: Model for Google (default: gemini-2.0-flash)
- GOOGLE_API_KEY: Required for Google provider
- OLLAMA_HOST: Optional Ollama host (default: localhost)
- OLLAMA_PORT: Optional Ollama port (default: 11434)

Usage:
    python -m examples.unified_eval                        # Uses default provider (Ollama)
    LLM_PROVIDER=anthropic python -m examples.unified_eval # Uses Anthropic
    LLM_PROVIDER=openai python -m examples.unified_eval    # Uses OpenAI
    LLM_PROVIDER=google python -m examples.unified_eval    # Uses Google Gemini
"""

import logging
import json
from pathlib import Path

from src.llm_eval.core import evaluator, config
from src.llm_eval.adapters import AnthropicAdapter, GoogleAdapter, OllamaAdapter, OpenAIAdapter

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
    elif provider == "anthropic":
        logger.info(f"Using Anthropic provider with model '{config.anthropic_model}'")
        return AnthropicAdapter(
            model=config.anthropic_model,
            api_key=config.anthropic_api_key,
        )
    elif provider == "openai":
        logger.info(f"Using OpenAI provider with model '{config.openai_model}'")
        return OpenAIAdapter(
            model=config.openai_model,
            api_key=config.openai_api_key,
        )
    elif provider == "google":
        logger.info(f"Using Google provider with model '{config.google_model}'")
        return GoogleAdapter(
            model=config.google_model,
            api_key=config.google_api_key,
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider}. "
            f"Valid options are: 'ollama', 'anthropic', 'openai', 'google'"
        )


def evaluate():
    """Run evaluation with the configured LLM provider."""
    adapter = create_adapter()
    
    logger.info(f"Loading test data from {DATA_FILE}")
    
    # Load the test data structure
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    
    # Extract base data and create test cases from images
    base_data = {k: v for k, v in data.items() if k != "images"}
    test_cases = [
        (evaluator.generate_question(base_data, image_item), image_item["expected"])
        for image_item in data["images"]
    ]
    
    logger.info(f"Loaded {len(test_cases)} test cases")
    evaluator.simple_exact_match_eval(test_cases, adapter.ask, model_name=adapter.model_name)


if __name__ == "__main__":
    evaluate()