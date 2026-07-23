# llm-eval

Project for evaluating image-based LLM/evaluation pipelines.

llm-eval/
├── src/                    # Core library
│   ├── llm_eval/           # Evaluator implementation
│   ├── templates/          # Question/prompt templates
│   └── utils/              # Helper utilities
├── examples/               # Demo scripts and usage examples
├── generate_samples.py     # Script to generate sample test data
└── tests/                  # Unit tests

## Configuration

The evaluator uses environment variables for configuration. See `.env.example` for details.

## Test Data Management

Test data is stored in `/data/` as JSON files — each file is a single document with shared metadata (`count`, `attribute`, `options`) plus a list of individual test cases (`images`). Both OpenAI and Ollama evaluators load the same file to enable fair model comparison.

### Data Files

- `data/car_color_samples.json` — test cases for car color classification used by both Ollama and OpenAI evaluators (for direct comparison)
- `data/shoe_color_samples.json` — test cases for shoe color classification

### File Structure

Each test data file contains top-level metadata that applies to all test cases in the file, plus an `images` array with per-sample entries:

```json
{
  "count": 1,
  "attribute": "car colour",
  "options": ["red", "blue", "grey", "white", "black", "yellow"],
  "images": [
    { "expected": "red", "url": "https://example.com/image.jpg" }
  ]
}
```

- `count`, `attribute`, `options` are shared across every test case in the file
- `images` is the list of individual test cases, each with its own `url` and `expected` answer

### Adding More Samples

**Option 1: Edit the JSON file directly** — append an entry to the `images` array.

**Option 2: Load, modify, and save with DataLoader**

```python
import json
from src.llm_eval.data import DataLoader

# Load file
data = DataLoader.load_json("data/car_color_samples.json")
# Append a new test case
data["images"].append({"url": "https://example.com/new.jpg", "expected": "red"})
# Save back
with open("data/car_color_samples.json", "w") as f:
    json.dump(data, f, indent=2)
```

### Using the DataLoader

```python
from src.llm_eval.data import DataLoader

# Load the JSON file
data = DataLoader.load_json("data/car_color_samples.json")
# Separate base metadata from test cases
base_data = {k: v for k, v in data.items() if k != "images"}
test_cases = data["images"]
```

### Best Practices for Test Data

1. **Version your data** — tag datasets with dates or version numbers
2. **Make data regenerable** — script your data generation (not hand-curated)
3. **Organize by split** — keep train/test/val separate or tagged
4. **Document schema** — add comments in your data files about expected fields
5. **Consider data privacy** — don't commit sensitive test data to git

## Extending the Framework

Extend `src/llm_eval/core/evaluator.py` with new evaluation functions.

Add new JSON files to `src/llm_eval/templates/` and update the evaluator to support them.

## Testing

Run the full test suite:

```bash
python -m pytest tests/ -v
```

Tests cover the evaluator logic (question generation, accuracy calculation) and data loading (JSON read). No API keys required — all tests use mocks and temporary files.

## Troubleshooting

If you get a `ConnectionError` when running the Ollama evaluator:

- Ensure the Ollama daemon is running and reachable at the configured host/port.
- Check any required API keys or environment variables are set.
- Run tests with network calls mocked when possible to avoid external dependencies.
