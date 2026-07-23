"""
Unified LLM evaluator that works with any provider (Ollama, OpenAI, etc.).

The provider and model are configured via environment variables:
- LLM_PROVIDER: 'ollama' (default) or 'openai'
- OLLAMA_MODEL: Model for Ollama (default: gemma3:1b)
- OPENAI_MODEL: Model for OpenAI (default: gpt-4o-mini)
- OPENAI_API_KEY: Required for OpenAI provider
- USE_LLM_JUDGE: 'true' to additionally run LLM-as-judge evaluation (default: false)

Usage:
    python -m examples.unified_eval              # Uses default provider (Ollama)
    LLM_PROVIDER=openai python -m examples.unified_eval  # Uses OpenAI
"""

import logging
from pathlib import Path

from llm_eval.core import config, evaluator, judge
from llm_eval.data import load_json
from llm_eval.adapters import OllamaAdapter, OpenAIAdapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load test data from external file
DATA_FILE = Path(__file__).parent.parent / "data" / "review_aspects_samples.json"


def create_adapter():
    """Create the appropriate LLM adapter based on configuration."""
    provider = config.llm_provider

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

    data = load_json(str(DATA_FILE))

    # Extract base data and create test cases from reviews
    base_data = {k: v for k, v in data.items() if k != "reviews"}
    test_cases = [
        (evaluator.generate_question(base_data, review_item), review_item["expected"])
        for review_item in data["reviews"]
    ]

    logger.info(f"Loaded {len(test_cases)} test cases")

    # Ask the model once per test case; both evaluators below score these same answers.
    answers = evaluator.collect_answers(test_cases, adapter.ask)

    evaluator.multi_label_eval(answers, options=base_data["options"], model_name=adapter.model_name)

    if config.use_llm_judge:
        logger.info("USE_LLM_JUDGE=true: running LLM-as-judge evaluation")
        judge.llm_judge_eval(
            answers,
            judge_llm=adapter.ask,
            options=base_data["options"],
            model_name=adapter.model_name,
        )


if __name__ == "__main__":
    evaluate()
