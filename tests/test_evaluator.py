import pytest
from src.llm_eval.core import evaluator


class TestGenerateQuestion:
    def setup_method(self):
        self.base_data = {
            "attribute": "aspects",
            "options": ["price", "quality", "shipping", "customer service"],
        }
        self.review_item = {
            "text": "Great value but it broke after a week.",
            "expected": ["price", "quality"],
        }

    def test_generates_question_with_valid_data(self):
        question = evaluator.generate_question(self.base_data, self.review_item)
        assert "price" in question
        assert "quality" in question
        assert "shipping" in question
        assert "customer service" in question
        assert "Great value but it broke after a week." in question

    @pytest.mark.parametrize(
        "missing_from, expected_match",
        [
            ("base_data", "Missing required keys in base data"),
            ("review_item", "Missing required keys in review item"),
        ],
    )
    def test_missing_required_keys(self, missing_from, expected_match):
        kwargs = {"base_data": self.base_data, "review_item": self.review_item}
        kwargs[missing_from] = {}
        with pytest.raises(KeyError, match=expected_match):
            evaluator.generate_question(**kwargs)

    @pytest.mark.parametrize("invalid_options", ["not a list", 123, None, {"a": 1}])
    def test_invalid_options_type_raises(self, invalid_options):
        self.base_data["options"] = invalid_options
        with pytest.raises(ValueError, match="'options' must be a list or tuple"):
            evaluator.generate_question(self.base_data, self.review_item)

    @pytest.mark.parametrize(
        "options, expected_substring",
        [
            (["price", "quality", "shipping"], "price, quality, shipping"),
            (("price", "quality"), "price, quality"),
            (["price"], "price"),
        ],
        ids=["list", "tuple", "single_item"],
    )
    def test_options_formatting(self, options, expected_substring):
        self.base_data["options"] = options
        question = evaluator.generate_question(self.base_data, self.review_item)
        assert expected_substring in question


class TestMultiLabelEval:
    OPTIONS = ["price", "quality", "shipping", "customer service", "packaging"]

    @pytest.mark.parametrize(
        "expected, answer, expected_f1",
        [
            (["price", "quality"], "This review mentions price and quality.", 1.0),
            (["price"], "The shipping was great.", 0.0),
            (["price", "quality", "shipping"], "Mentions price only.", 0.5),
            ([], "Everything about this purchase was perfect, no complaints at all.", 1.0),
            ([], "The price was way too high.", 0.0),
            (["Price"], "The PRICE is too high.", 1.0),
        ],
        ids=[
            "all_correct",
            "all_wrong",
            "partial_match",
            "both_empty",
            "false_positive",
            "case_insensitive",
        ],
    )
    def test_f1_scoring(self, expected, answer, expected_f1):
        test_cases = [("Q1", expected)]
        f1 = evaluator.multi_label_eval(
            test_cases, lambda q: answer, options=self.OPTIONS, model_name="mock-model"
        )
        assert f1 == pytest.approx(expected_f1)

    def test_empty_test_cases(self):
        f1 = evaluator.multi_label_eval(
            [], lambda q: "anything", options=self.OPTIONS, model_name="mock-model"
        )
        assert f1 == 0.0
