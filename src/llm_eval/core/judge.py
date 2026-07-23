"""LLM-as-judge evaluation.

Substring matching (see `evaluator.multi_label_eval`) is fast and free but
brittle: a model that answers "cost" instead of "price" is marked wrong even
though it identified the right aspect. This module asks a second LLM (the
"judge") to assess correctness in natural language instead, trading cost and
latency for tolerance to paraphrasing.
"""

from typing import Callable, List, Tuple

from ._template_loader import TEMPLATES_DIR, load_template


def _parse_verdict(judge_response: str) -> bool:
    """Extract the correct/incorrect verdict from the judge's free-text response."""
    for line in judge_response.splitlines():
        stripped = line.strip().upper()
        if stripped.startswith("VERDICT:"):
            return "INCORRECT" not in stripped and "CORRECT" in stripped

    # Judge didn't follow the requested format; fall back to a loose keyword check.
    lowered = judge_response.lower()
    return "incorrect" not in lowered and "correct" in lowered


def llm_judge_eval(
    answers: List[Tuple[str, str, List[str]]],
    judge_llm: Callable[[str], str],
    options: List[str],
    model_name: str = "",
) -> float:
    """Judge (question, answer, expected_labels) triples with an LLM instead of substring matching.

    Use `collect_answers` (from `evaluator.py`) to build `answers` — the same
    triples can be scored by both `multi_label_eval` and this function
    without asking the model twice. For each triple, asks `judge_llm`
    whether the answer correctly and completely identifies the expected
    aspects. Prints a per-example report and returns the fraction of examples
    judged correct.
    """
    header = f"LLM-as-judge evaluation: {model_name}" if model_name else "LLM-as-judge evaluation"
    print(f"{header}\n")

    options_str = ", ".join(options)
    template = load_template(TEMPLATES_DIR / "judge_aspects.json", "judge_template")

    correct = 0
    for question, answer, expected in answers:
        expected_str = ", ".join(expected) if expected else "none"

        judge_prompt = template.format(
            question=question,
            model_answer=answer,
            expected_str=expected_str,
            options_str=options_str,
        )
        verdict_response = judge_llm(judge_prompt)
        is_correct = _parse_verdict(verdict_response)

        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Expected: {expected_str}")
        print(f"Judge verdict: {'correct' if is_correct else 'incorrect'}")
        print("-" * 70)

        if is_correct:
            correct += 1

    accuracy = correct / len(answers) if answers else 0.0
    print(f"\nJudge accuracy: {accuracy:.2f}")
    return accuracy
