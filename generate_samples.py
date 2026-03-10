"""
Example script to generate test data programmatically.

This demonstrates how to create and save test datasets for evaluation.
Both OpenAI and Ollama evaluators use the same dataset for fair comparison.
"""

from Test.data_loader import DataLoader


def generate_samples(count: int = 100) -> list:
    """Generate synthetic test samples in ollama format.
    
    Both Ollama and OpenAI evaluators use this format for comparison.
    """
    sample_colors = [
        ("red", "car"),
        ("blue", "car"),
        ("green", "car"),
        ("yellow", "car"),
        ("black", "car"),
    ]
    
    samples = []
    for i in range(count):
        color_expected, obj_type = sample_colors[i % len(sample_colors)]
        samples.append({
            "count": 1,
            "attribute": f"{obj_type} colour",
            "options": ["red", "blue", "yellow", "green", "black"],
            "url": f"https://example.com/image_{i}.jpg",
            "expected": color_expected,
            "difficulty": "easy",
            "id": i + 1
        })
    
    return samples


if __name__ == "__main__":
    # Generate and save samples (used by both Ollama and OpenAI)
    print("Generating test samples (used by both evaluators)...")
    samples = generate_samples(100)
    DataLoader.save_jsonl(samples, "data/car_color_samples_expanded.jsonl")
    print(f"✓ Saved {len(samples)} samples to data/car_color_samples_expanded.jsonl")
    print("\nTo use the expanded dataset, update the DATA_FILE path in eval_ollama.py/eval_openai.py")
    print("Both evaluators will use the same data for fair comparison.")
