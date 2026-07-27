"""Build a small, our-schema sample from the real ABSA dataset explored in
explore_absa_dataset.py, so it can be run through the actual eval pipeline
(examples/unified_eval.py) instead of just analyzed.

Deliberately NOT stratified by category -- a random sample preserves the
real class imbalance found during exploration (food/anecdotes dominate,
price is rare), which is worth seeing reflected in the eval's per-label
support counts rather than papered over. Exact-duplicate review text found
during exploration is dropped before sampling, so we don't carry a known
hygiene issue into a dataset we're presenting as curated.

Usage:
    pip install -e ".[analysis]"
    python -m scripts.build_absa_sample
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "scholl99/absa-restaurant-processed-v1"
SAMPLE_SIZE = 200
SEED = 42
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "absa_restaurant_samples.json"

# Confirmed via scripts/explore_absa_dataset.py -- the label set is exactly these 5.
OPTIONS = ["food", "service", "ambience", "price", "anecdotes/miscellaneous"]


def to_categories(labels: list) -> list:
    """Convert "category#polarity" strings into a deduped, order-preserving list of categories."""
    seen = []
    for label in labels:
        category = label.partition("#")[0]
        if category not in seen:
            seen.append(category)
    return seen


def main():
    ds = load_dataset(DATASET_ID)["train"]

    seen_texts = set()
    deduped = []
    for row in ds:
        if row["text"] not in seen_texts:
            seen_texts.add(row["text"])
            deduped.append(row)

    random.seed(SEED)
    sample = random.sample(deduped, SAMPLE_SIZE)

    reviews = [{"text": row["text"], "expected": to_categories(row["label"])} for row in sample]

    data = {
        "version": "1.1",
        "updated": "2026-07-25",
        "source": (
            f"Random sample of {SAMPLE_SIZE} (seed={SEED}), deduplicated, from Hugging "
            f"Face dataset '{DATASET_ID}' (SemEval-2014 Restaurant ABSA). Polarity labels "
            f"dropped -- only aspect category kept, to match our aspect-mention schema."
        ),
        "attribute": "aspects",
        "options": OPTIONS,
        "reviews": reviews,
    }

    OUTPUT_FILE.write_text(json.dumps(data, indent=2))
    print(f"Wrote {len(reviews)} reviews to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
