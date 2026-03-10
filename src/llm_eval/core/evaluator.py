import json
from pathlib import Path
from typing import Callable, List, Tuple, Dict, Optional


# Lazy-load question template on first use
_QUESTION_TEMPLATE: Optional[str] = None


def _load_question_template() -> str:
    """Load the question template from configuration file (lazy-loaded)."""
    global _QUESTION_TEMPLATE
    if _QUESTION_TEMPLATE is not None:
        return _QUESTION_TEMPLATE
    
    config_file = Path(__file__).parent.parent / "templates" / "car_color.json"
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        _QUESTION_TEMPLATE = config.get("question_template", "")
        if not _QUESTION_TEMPLATE:
            raise ValueError("question_template key not found in JSON config")
        return _QUESTION_TEMPLATE
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Question template config not found at {config_file}. "
            "Please ensure car_color.json exists in the templates directory."
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in car_color.json: {e}")


def generate_question(data: Dict) -> str:
    """Generate a question from test data parameters using the configured template.
    
    The template is loaded from car_color.json and can be customized
    by editing that file without modifying this code.
    
    Args:
        data: Dictionary with keys: count, attribute, options, url, expected
        
    Raises:
        KeyError: If required keys are missing from data
        ValueError: If data types are invalid
    """
    # Validate required keys
    required_keys = {"count", "attribute", "options", "url", "expected"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise KeyError(f"Missing required keys in test data: {missing_keys}")
    
    # Validate data types
    if not isinstance(data["options"], (list, tuple)):
        raise ValueError("'options' must be a list or tuple")
    
    count = data["count"]
    attribute = data["attribute"]
    options = data["options"]
    url = data["url"]
    options_str = ", ".join(str(opt) for opt in options)
    
    template = _load_question_template()  # Lazy load on first use
    return template.format(
        count=count,
        attribute=attribute,
        options_str=options_str,
        url=url
    )


def simple_exact_match_eval(
    test_cases: List[Tuple[str, str]],
    ask_llm: Callable[[str], str],
    model_name: str = "",
) -> float:
    """Evaluate a list of (question, expected) pairs against an LLM.

    The `ask_llm` callable should accept a question string and return the
    model's answer. Returns accuracy as a float between 0.0 and 1.0 and
    also prints a human‑readable report.
    """
    correct = 0
    total = len(test_cases)

    header = f"Testing the model: {model_name}" if model_name else "Evaluating model"
    print(f"{header}\n")

    for question, expected in test_cases:
        answer = ask_llm(question)
        is_correct = expected.lower() in answer.lower()
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Expected: {expected}  →  {'✓ Correct' if is_correct else '✗ Wrong'}")
        print("-" * 70)
        if is_correct:
            correct += 1

    accuracy = correct / total if total else 0.0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.0%}")
    return accuracy