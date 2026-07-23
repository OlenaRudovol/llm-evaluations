import pytest
from src.llm_eval.core import evaluator


class TestGenerateQuestion:
    def setup_method(self):
        self.base_data = {
            "count": 1,
            "attribute": "car colour",
            "options": ["red", "blue", "green", "yellow"],
        }
        self.image_item = {
            "url": "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2",
            "expected": "yellow",
        }

    def test_generates_question_with_valid_data(self):
        question = evaluator.generate_question(self.base_data, self.image_item)
        assert "car colour" in question
        assert "blue" in question
        assert "red" in question
        assert "green" in question
        assert "yellow" in question
        assert "https://images.unsplash.com" in question

    @pytest.mark.parametrize(
        "missing_from, expected_match",
        [
            ("base_data", "Missing required keys in base data"),
            ("image_item", "Missing required keys in image item"),
        ],
    )
    def test_missing_required_keys(self, missing_from, expected_match):
        kwargs = {"base_data": self.base_data, "image_item": self.image_item}
        kwargs[missing_from] = {}
        with pytest.raises(KeyError, match=expected_match):
            evaluator.generate_question(**kwargs)

    @pytest.mark.parametrize("invalid_options", ["not a list", 123, None, {"a": 1}])
    def test_invalid_options_type_raises(self, invalid_options):
        self.base_data["options"] = invalid_options
        with pytest.raises(ValueError, match="'options' must be a list or tuple"):
            evaluator.generate_question(self.base_data, self.image_item)

    @pytest.mark.parametrize(
        "options, expected_substring",
        [
            (["red", "blue", "green"], "red, blue, green"),
            (("red", "blue"), "red, blue"),
            (["red"], "red"),
        ],
        ids=["list", "tuple", "single_item"],
    )
    def test_options_formatting(self, options, expected_substring):
        self.base_data["options"] = options
        question = evaluator.generate_question(self.base_data, self.image_item)
        assert expected_substring in question


class TestSimpleExactMatchEval:
    @pytest.mark.parametrize(
        "test_cases, answers, expected_accuracy",
        [
            ([("Q1", "red"), ("Q2", "blue")], ["red", "blue"], 1.0),
            ([("Q1", "red"), ("Q2", "blue")], ["green", "green"], 0.0),
            ([("Q1", "red"), ("Q2", "blue")], ["red", "green"], 0.5),
            ([("Q1", "red"), ("Q2", "blue"), ("Q3", "green")], ["red", "x", "green"], 2 / 3),
            ([("Q1", "Red")], ["RED"], 1.0),
            ([("Q1", "red")], ["The color is red I think"], 1.0),
            ([], [], 0.0),
        ],
        ids=[
            "all_correct",
            "all_wrong",
            "partial_match",
            "two_of_three",
            "case_insensitive",
            "substring_match",
            "empty_test_cases",
        ],
    )
    def test_accuracy_calculation(self, test_cases, answers, expected_accuracy):
        answers_iter = iter(answers)
        accuracy = evaluator.simple_exact_match_eval(
            test_cases, lambda q: next(answers_iter), model_name="mock-model"
        )
        assert accuracy == pytest.approx(expected_accuracy)
