from llm_eval.core import judge


class TestLlmJudgeEval:
    def test_all_correct(self):
        answers = [("Q1", "some answer", ["price"]), ("Q2", "some answer", ["quality"])]
        accuracy = judge.llm_judge_eval(
            answers,
            judge_llm=lambda p: "Looks right.\nVERDICT: correct",
            model_name="mock-model",
        )
        assert accuracy == 1.0

    def test_all_incorrect(self):
        answers = [("Q1", "some answer", ["price"])]
        accuracy = judge.llm_judge_eval(
            answers,
            judge_llm=lambda p: "Missing an aspect.\nVERDICT: incorrect",
        )
        assert accuracy == 0.0

    def test_mixed_verdicts(self):
        answers = [("Q1", "answer", ["price"]), ("Q2", "answer", ["quality"])]
        responses = iter(["VERDICT: correct", "VERDICT: incorrect"])
        accuracy = judge.llm_judge_eval(
            answers,
            judge_llm=lambda p: next(responses),
        )
        assert accuracy == 0.5

    def test_empty_answers(self):
        accuracy = judge.llm_judge_eval(
            [],
            judge_llm=lambda p: "VERDICT: correct",
        )
        assert accuracy == 0.0

    def test_verdict_fallback_without_exact_format(self):
        # Judge didn't follow the requested "VERDICT: correct|incorrect" line format.
        answers = [("Q1", "answer", ["price"])]
        accuracy = judge.llm_judge_eval(
            answers,
            judge_llm=lambda p: "Yes, this looks correct to me.",
        )
        assert accuracy == 1.0

    def test_prompt_includes_question_answer_and_expected(self):
        captured = {}

        def judge_llm(prompt):
            captured["prompt"] = prompt
            return "VERDICT: correct"

        answers = [("What aspects does this mention?", "It's about the price.", ["price"])]
        judge.llm_judge_eval(answers, judge_llm=judge_llm)

        assert "What aspects does this mention?" in captured["prompt"]
        assert "It's about the price." in captured["prompt"]
        assert "price" in captured["prompt"]

    def test_prompt_omits_full_options_list(self):
        # The judge previously saw the full options list alongside the expected
        # subset and would sometimes mark a correct answer "incorrect" for not
        # covering every possible aspect. The prompt should only ever mention
        # the actual expected labels.
        captured = {}

        def judge_llm(prompt):
            captured["prompt"] = prompt
            return "VERDICT: correct"

        answers = [("Which aspects: price, quality, shipping?", "price", ["price"])]
        judge.llm_judge_eval(answers, judge_llm=judge_llm)

        assert "ONLY correct aspects" in captured["prompt"]
        assert "all possible aspects" not in captured["prompt"]
