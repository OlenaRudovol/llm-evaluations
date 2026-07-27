"""Integrity checks for the real datasets shipped in data/.

Unlike the other test files, these intentionally load the actual data files
rather than synthetic fixtures — the thing under test here is whether each
dataset is internally consistent, not a function's logic. Runs against every
*.json file in data/, so a new dataset is covered automatically.
"""

from pathlib import Path

import pytest

from llm_eval.data import load_json

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_FILES = sorted(DATA_DIR.glob("*.json"))


@pytest.fixture(params=DATA_FILES, ids=[f.name for f in DATA_FILES])
def dataset(request):
    return load_json(str(request.param))


class TestDatasetIntegrity:
    def test_every_expected_label_is_a_known_option(self, dataset):
        # `options` and each review's `expected` are edited independently in the
        # JSON file — nothing enforces they stay in sync. A label used in
        # `expected` but missing from `options` is never offered to the model
        # (it can't possibly predict it) yet still counts against every
        # aggregate metric, and it silently disappears from the per-label
        # breakdown since that iterates over `options`. Catch that drift here.
        options = {opt.lower() for opt in dataset["options"]}
        for review in dataset["reviews"]:
            unknown = {label.lower() for label in review["expected"]} - options
            assert not unknown, f"Review {review['text']!r} expects unknown label(s): {unknown}"

    def test_options_has_no_duplicate_entries(self, dataset):
        options = [opt.lower() for opt in dataset["options"]]
        assert len(options) == len(set(options)), f"Duplicate entries in options: {options}"

    def test_no_duplicate_review_text(self, dataset):
        # Duplicate reviews inflate the dataset without adding new signal —
        # the same trap as duplicate test cases in a regression suite.
        texts = [review["text"] for review in dataset["reviews"]]
        assert len(texts) == len(set(texts)), "Duplicate review text found in dataset"

    def test_has_version_metadata(self, dataset):
        # So eval results can be tied to a specific dataset version instead of
        # silently comparing runs against different underlying data.
        assert dataset.get("version"), "Dataset is missing a 'version' field"
        assert dataset.get("updated"), "Dataset is missing an 'updated' field"
