import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# Lazy-load question template on first use
_QUESTION_TEMPLATE: Optional[str] = None


def _load_question_template() -> str:
    """Load the question template from configuration file (lazy-loaded)."""
    global _QUESTION_TEMPLATE
    if _QUESTION_TEMPLATE is not None:
        return _QUESTION_TEMPLATE

    config_file = Path(__file__).parent.parent / "templates" / "review_aspects.json"
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
            "Please ensure review_aspects.json exists in the templates directory."
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in review_aspects.json: {e}")


def generate_question(base_data: Dict, review_item: Dict) -> str:
    """Generate a question from test data parameters using the configured template.

    The template is loaded from review_aspects.json and can be customized
    by editing that file without modifying this code.

    Args:
        base_data: Dictionary with keys: attribute, options
        review_item: Dictionary with keys: text, expected

    Raises:
        KeyError: If required keys are missing from data
        ValueError: If data types are invalid
    """
    # Validate required keys
    base_required = {"attribute", "options"}
    review_required = {"text", "expected"}

    base_missing = base_required - set(base_data.keys())
    review_missing = review_required - set(review_item.keys())

    if base_missing:
        raise KeyError(f"Missing required keys in base data: {base_missing}")
    if review_missing:
        raise KeyError(f"Missing required keys in review item: {review_missing}")

    # Validate data types
    if not isinstance(base_data["options"], (list, tuple)):
        raise ValueError("'options' must be a list or tuple")

    attribute = base_data["attribute"]
    options = base_data["options"]
    text = review_item["text"]
    options_str = ", ".join(str(opt) for opt in options)

    template = _load_question_template()  # Lazy load on first use
    return template.format(
        attribute=attribute,
        options_str=options_str,
        text=text,
    )


def multi_label_eval(
    test_cases: List[Tuple[str, List[str]]],
    ask_llm: Callable[[str], str],
    options: List[str],
    model_name: str = "",
) -> float:
    """Evaluate a list of (question, expected_labels) pairs against an LLM.

    The `ask_llm` callable should accept a question string and return the
    model's free-text answer. Labels are detected by checking which of the
    known `options` appear (case-insensitively) as a substring of that
    answer. Returns the average per-example F1 score across all test cases
    and also prints a human-readable report.
    """
    header = f"Testing the model: {model_name}" if model_name else "Evaluating model"
    print(f"{header}\n")

    total_f1 = 0.0
    for question, expected in test_cases:
        answer = ask_llm(question)
        answer_lower = answer.lower()

        predicted = {opt.lower() for opt in options if opt.lower() in answer_lower}
        expected_set = {label.lower() for label in expected}

        if not predicted and not expected_set:
            f1 = 1.0
        elif not predicted or not expected_set:
            f1 = 0.0
        else:
            overlap = predicted & expected_set
            precision = len(overlap) / len(predicted)
            recall = len(overlap) / len(expected_set)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Predicted: {sorted(predicted) or ['none']}")
        print(f"Expected:  {sorted(expected_set) or ['none']}")
        print(f"F1: {f1:.2f}")
        print("-" * 70)

        total_f1 += f1

    avg_f1 = total_f1 / len(test_cases) if test_cases else 0.0
    print(f"\nAverage F1: {avg_f1:.2f}")
    return avg_f1
