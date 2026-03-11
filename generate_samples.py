"""
Example script to generate test data programmatically.

This demonstrates how to create and save test datasets for evaluation.
Both OpenAI and Ollama evaluators use the same dataset for fair comparison.
"""

import json
from pathlib import Path


def generate_samples(count: int = 100) -> dict:
    """Generate synthetic test samples with shared metadata and images array.
    
    Returns a structured dataset with shared base data and individual image items.
    """
    sample_colors = [
        ("red", "car"),
        ("blue", "car"),
        ("green", "car"),
        ("yellow", "car"),
        ("black", "car"),
    ]
    
    images = []
    for i in range(count):
        color_expected, obj_type = sample_colors[i % len(sample_colors)]
        images.append({
            "url": f"https://example.com/image_{i}.jpg",
            "expected": color_expected
        })
    
    return {
        "count": 1,
        "attribute": "car colour",
        "options": ["red", "blue", "yellow", "green", "black"],
        "images": images
    }


if __name__ == "__main__":
    # Generate and save samples
    print("Generating test samples (used by both evaluators)...")
    data = generate_samples(100)
    
    output_path = Path("data/car_color_samples_expanded.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved {len(data['images'])} samples to {output_path}")
    print("\nTo use the expanded dataset, update the DATA_FILE path in examples/unified_eval.py")
    print("Both evaluators will use the same data for fair comparison.")
