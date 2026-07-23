from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

from ._template_loader import load_template

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


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

    template = load_template(TEMPLATES_DIR / "review_aspects.json", "question_template")
    return template.format(
        attribute=attribute,
        options_str=options_str,
        text=text,
    )


def collect_answers(
    test_cases: List[Tuple[str, List[str]]],
    ask_llm: Callable[[str], str],
) -> List[Tuple[str, str, List[str]]]:
    """Ask the model each question once, pairing it with its answer and expected labels.

    Call this once and pass the result to `multi_label_eval` and/or
    `llm_judge_eval` so each test case is only sent to the model a single
    time, even when running multiple evaluation methods on the same data.
    """
    return [(question, ask_llm(question), expected) for question, expected in test_cases]


def exact_match_eval(pairs: List[Tuple[Set[str], Set[str]]]) -> float:
    """Fraction of examples where the predicted label set exactly equals the expected set.

    Stricter than per-example F1: a partially-correct prediction counts as a miss.
    """
    if not pairs:
        return 0.0
    matches = sum(1 for predicted, expected in pairs if predicted == expected)
    return matches / len(pairs)


def micro_prf1_eval(pairs: List[Tuple[Set[str], Set[str]]]) -> Dict[str, float]:
    """Micro-averaged precision/recall/F1, aggregating TP/FP/FN across all examples.

    Unlike the per-example (macro) F1 in `multi_label_eval`, this weights every
    individual label decision equally rather than every example equally.
    """
    tp = fp = fn = 0
    for predicted, expected in pairs:
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def multi_label_eval(
    answers: List[Tuple[str, str, List[str]]],
    options: List[str],
    model_name: str = "",
) -> Dict[str, float]:
    """Score (question, answer, expected_labels) triples via substring matching.

    Use `collect_answers` to build `answers`. Labels are detected by checking
    which of the known `options` appear (case-insensitively) as a substring
    of the model's answer. Prints a human-readable report and returns a dict
    with `avg_f1` (macro, per-example), `exact_match`, and `micro`
    (a `{precision, recall, f1}` dict).
    """
    header = f"Testing the model: {model_name}" if model_name else "Evaluating model"
    print(f"{header}\n")

    pairs: List[Tuple[Set[str], Set[str]]] = []
    total_f1 = 0.0
    for question, answer, expected in answers:
        answer_lower = answer.lower()

        predicted = {opt.lower() for opt in options if opt.lower() in answer_lower}
        expected_set = {label.lower() for label in expected}
        pairs.append((predicted, expected_set))

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

    avg_f1 = total_f1 / len(answers) if answers else 0.0
    exact_match = exact_match_eval(pairs)
    micro = micro_prf1_eval(pairs)

    print(f"\nAverage F1 (macro, per-example): {avg_f1:.2f}")
    print(f"Exact match rate:                {exact_match:.2f}")
    print(
        f"Micro precision / recall / F1:   "
        f"{micro['precision']:.2f} / {micro['recall']:.2f} / {micro['f1']:.2f}"
    )

    return {"avg_f1": avg_f1, "exact_match": exact_match, "micro": micro}
