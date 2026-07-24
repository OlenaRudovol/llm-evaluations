"""Integrity checks for the real dataset shipped in data/.

Unlike the other test files, these intentionally load the actual production
data file rather than synthetic fixtures — the thing under test here is
whether the dataset itself is internally consistent, not a function's logic.
"""

from pathlib import Path

from llm_eval.data import load_json

DATA_FILE = Path(__file__).parent.parent / "data" / "review_aspects_samples.json"


class TestReviewAspectsSamplesIntegrity:
    def setup_method(self):
        self.data = load_json(str(DATA_FILE))

    def test_every_expected_label_is_a_known_option(self):
        # `options` and each review's `expected` are edited independently in the
        # JSON file — nothing enforces they stay in sync. A label used in
        # `expected` but missing from `options` is never offered to the model
        # (it can't possibly predict it) yet still counts against every
        # aggregate metric, and it silently disappears from the per-label
        # breakdown since that iterates over `options`. Catch that drift here.
        options = {opt.lower() for opt in self.data["options"]}
        for review in self.data["reviews"]:
            unknown = {label.lower() for label in review["expected"]} - options
            assert not unknown, f"Review {review['text']!r} expects unknown label(s): {unknown}"

    def test_options_has_no_duplicate_entries(self):
        options = [opt.lower() for opt in self.data["options"]]
        assert len(options) == len(set(options)), f"Duplicate entries in options: {options}"

    def test_no_duplicate_review_text(self):
        # Duplicate reviews inflate the dataset without adding new signal —
        # the same trap as duplicate test cases in a regression suite.
        texts = [review["text"] for review in self.data["reviews"]]
        assert len(texts) == len(set(texts)), "Duplicate review text found in dataset"

    def test_has_version_metadata(self):
        # So eval results can be tied to a specific dataset version instead of
        # silently comparing runs against different underlying data.
        assert self.data.get("version"), "Dataset is missing a 'version' field"
        assert self.data.get("updated"), "Dataset is missing an 'updated' field"
