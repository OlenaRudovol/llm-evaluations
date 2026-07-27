"""
Unified LLM evaluator that works with any provider (Ollama, OpenAI, etc.).

The provider and model are configured via environment variables:
- LLM_PROVIDER: 'ollama' (default), 'openai', or 'groq'
- OLLAMA_MODEL: Model for Ollama (default: gemma3:1b)
- OPENAI_MODEL: Model for OpenAI (default: gpt-4o-mini)
- OPENAI_API_KEY: Required for OpenAI provider
- GROQ_MODEL: Model for Groq (default: llama-3.1-8b-instant)
- GROQ_API_KEY: Required for Groq provider (free tier at console.groq.com)
- USE_LLM_JUDGE: 'true' to additionally run LLM-as-judge evaluation (default: false)
- JUDGE_MODEL: Model to use as the judge, same provider as LLM_PROVIDER (default:
  unset, meaning the judge is the same model being tested — a weak/fast model can
  be an unreliable judge of its own answers, so overriding this with a stronger
  model on the same provider often gives more trustworthy verdicts)
- EVAL_DATA_FILE: Path to a data/*.json file to evaluate against (default:
  data/review_aspects_samples.json). All files in data/ share the same schema
  (attribute/options/reviews), so any of them works here.

Usage:
    python -m examples.unified_eval              # Uses default provider (Ollama)
    LLM_PROVIDER=openai python -m examples.unified_eval  # Uses OpenAI
    LLM_PROVIDER=groq python -m examples.unified_eval    # Uses Groq
    USE_LLM_JUDGE=true JUDGE_MODEL=llama-3.3-70b-versatile \\
        LLM_PROVIDER=groq python -m examples.unified_eval  # Stronger judge model
    EVAL_DATA_FILE=data/absa_restaurant_samples.json \\
        python -m examples.unified_eval  # Evaluate against a different dataset
"""

import logging
from pathlib import Path
from typing import Optional

from llm_eval.core import config, evaluator, judge
from llm_eval.data import load_json
from llm_eval.adapters import GroqAdapter, OllamaAdapter, OpenAIAdapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load test data from external file (override with EVAL_DATA_FILE)
DEFAULT_DATA_FILE = Path(__file__).parent.parent / "data" / "review_aspects_samples.json"
DATA_FILE = Path(config.data_file) if config.data_file else DEFAULT_DATA_FILE


def create_adapter(model: Optional[str] = None):
    """Create the appropriate LLM adapter based on configuration.

    Args:
        model: Override the provider's configured default model (used to run
            a different model — e.g. a stronger one — as the judge).
    """
    provider = config.llm_provider

    if provider == "ollama":
        model = model or config.ollama_model
        logger.info(f"Using Ollama provider with model '{model}'")
        return OllamaAdapter(model=model)
    elif provider == "openai":
        model = model or config.openai_model
        logger.info(f"Using OpenAI provider with model '{model}'")
        return OpenAIAdapter(model=model, api_key=config.openai_api_key)
    elif provider == "groq":
        model = model or config.groq_model
        logger.info(f"Using Groq provider with model '{model}'")
        return GroqAdapter(model=model, api_key=config.groq_api_key)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider}. "
            f"Valid options are: 'ollama', 'openai', 'groq'"
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
        judge_adapter = create_adapter(model=config.judge_model) if config.judge_model else adapter
        logger.info(f"USE_LLM_JUDGE=true: judging with model '{judge_adapter.model_name}'")
        judge.llm_judge_eval(
            answers,
            judge_llm=judge_adapter.ask,
            model_name=adapter.model_name,
        )


if __name__ == "__main__":
    evaluate()
