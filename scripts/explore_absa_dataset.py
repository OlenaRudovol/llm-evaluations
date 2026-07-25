"""One-off exploration: download a real ABSA dataset from Hugging Face and
practice the same data-hygiene checks we apply to our own hand-curated
dataset (data/review_aspects_samples.json) at a much larger, messier scale —
duplicates, label balance, missing values, and train/test leakage.

This is a throwaway analysis tool, not part of the eval pipeline, so its
dependency (`datasets`) is kept out of the main package requirements.

Usage:
    pip install -e ".[analysis]"
    python -m scripts.explore_absa_dataset
"""

from collections import Counter

from datasets import load_dataset

DATASET_ID = "scholl99/absa-restaurant-processed-v1"


def split_category_polarity(label: str) -> tuple:
    """"food#positive" -> ("food", "positive")."""
    category, _, polarity = label.partition("#")
    return category, polarity


def check_missing_fields(rows) -> None:
    """Flag rows with blank text or an empty label list -- both would silently
    break scoring downstream (an unlabeled review can't contribute to any
    label's precision/recall)."""
    empty_text = sum(1 for r in rows if not r["text"].strip())
    empty_labels = sum(1 for r in rows if not r["label"])
    print(f"Empty text: {empty_text}, empty label list: {empty_labels}")


def check_duplicate_text(rows) -> list:
    """Flag exact-duplicate review text -- inflates the dataset without adding
    signal. Returns the text list so the caller can reuse it for the
    cross-split leakage check instead of re-extracting it."""
    texts = [r["text"] for r in rows]
    duplicate_count = len(texts) - len(set(texts))
    print(f"Duplicate review texts: {duplicate_count} ({duplicate_count / len(texts):.1%})")
    return texts


def _print_distribution(counts: Counter) -> None:
    total = sum(counts.values())
    for key, count in counts.most_common():
        print(f"  {key:<25} {count:>5}  ({count / total:.1%})")


def report_category_and_polarity_distribution(rows) -> None:
    """Show how skewed the label set is -- e.g. a catch-all category or the
    majority sentiment dominating, which per-label metrics need `support` to
    contextualize (see evaluator.per_label_prf1)."""
    category_counts: Counter = Counter()
    polarity_counts: Counter = Counter()
    for r in rows:
        for label in r["label"]:
            category, polarity = split_category_polarity(label)
            category_counts[category] += 1
            polarity_counts[polarity] += 1

    print("\nCategory distribution:")
    _print_distribution(category_counts)

    print("\nPolarity distribution:")
    _print_distribution(polarity_counts)


def report_text_length(rows) -> None:
    """Show the spread of review length in words -- real data is rarely as
    uniform as a small hand-curated sample."""
    lengths = sorted(len(r["text"].split()) for r in rows)
    n = len(lengths)
    print(
        f"\nText length (words): min={lengths[0]}, median={lengths[n // 2]}, "
        f"max={lengths[-1]}, mean={sum(lengths) / n:.1f}"
    )


def report_labels_per_review(rows) -> None:
    """Show how multi-label the task actually is -- if almost every review has
    exactly one label, multi-label metrics add little over single-label ones."""
    label_count_dist: Counter = Counter(len(r["label"]) for r in rows)
    print("\nLabels per review:")
    for k in sorted(label_count_dist):
        print(f"  {k} label(s): {label_count_dist[k]} reviews")


def check_train_test_leakage(train_texts: list, test_texts: list) -> None:
    """Flag review text that appears verbatim in both splits -- if a model was
    fine-tuned on train, that overlap inflates its test score."""
    print(f"\n{'=' * 70}\nTrain/test leakage check\n{'=' * 70}")
    overlap = set(train_texts) & set(test_texts)
    print(f"Reviews appearing in both train and test: {len(overlap)}")
    if overlap:
        print("Example overlapping text:", repr(next(iter(overlap))[:100]))


def analyze_split(rows, split_name: str) -> list:
    """Run every per-split hygiene/analysis check and return the review texts
    (needed afterwards for the cross-split leakage check)."""
    print(f"\n{'=' * 70}\n{split_name} -- {len(rows)} rows\n{'=' * 70}")
    check_missing_fields(rows)
    texts = check_duplicate_text(rows)
    report_category_and_polarity_distribution(rows)
    report_text_length(rows)
    report_labels_per_review(rows)
    return texts


def main():
    ds = load_dataset(DATASET_ID)
    train_texts = analyze_split(ds["train"], "train")
    test_texts = analyze_split(ds["test"], "test")
    check_train_test_leakage(train_texts, test_texts)


if __name__ == "__main__":
    main()
