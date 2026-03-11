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


def generate_question(base_data: Dict, image_item: Dict) -> str:
    """Generate a question from test data parameters using the configured template.
    
    The template is loaded from car_color.json and can be customized
    by editing that file without modifying this code.
    
    Args:
        base_data: Dictionary with keys: count, attribute, options
        image_item: Dictionary with keys: url, expected
        
    Raises:
        KeyError: If required keys are missing from data
        ValueError: If data types are invalid
    """
    # Validate required keys
    base_required = {"count", "attribute", "options"}
    image_required = {"url", "expected"}
    
    base_missing = base_required - set(base_data.keys())
    image_missing = image_required - set(image_item.keys())
    
    if base_missing:
        raise KeyError(f"Missing required keys in base data: {base_missing}")
    if image_missing:
        raise KeyError(f"Missing required keys in image item: {image_missing}")
    
    # Validate data types
    if not isinstance(base_data["options"], (list, tuple)):
        raise ValueError("'options' must be a list or tuple")
    
    count = base_data["count"]
    attribute = base_data["attribute"]
    options = base_data["options"]
    url = image_item["url"]
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