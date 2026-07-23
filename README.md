# LLM Evaluation Framework

An open-source framework for evaluating Large Language Models (LLMs) with extensible adapters, templates, datasets, and metrics. Perfect for comparing model performance across different providers and tasks.

## Quick Start

1. **Install the package**:
   ```bash
   pip install -e .
   ```

2. **Set up environment variables**: Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

3. **Run the evaluator** with your preferred LLM provider:
   ```bash
   # Using default provider (Ollama)
   python -m examples.unified_eval
   
   # Or use OpenAI
   LLM_PROVIDER=openai python -m examples.unified_eval
   ```

4. If running locally, install Ollama (ollama.com), pull a model: `ollama pull gemma3:1b`

## Project Structure

```
llm-eval/
├── src/llm_eval/          # Core package
│   ├── adapters/          # LLM provider adapters (Ollama, OpenAI, etc.)
│   ├── core/              # Evaluation engine and configuration
│   ├── data/              # Data loading utilities
│   ├── templates/         # Question/prompt templates
│   └── utils/             # Helper utilities
├── examples/              # Demo scripts and usage examples
├── data/                  # Datasets
├── generate_samples.py    # Script to generate sample test data
└── tests/                 # Unit tests
```

## Configuration

The evaluator uses environment variables for configuration. See `.env.example` for all available options:

- `LLM_PROVIDER`: Which provider to use (default: `ollama`, options: `ollama`, `openai`)
- `OPENAI_API_KEY`: Your OpenAI API key (required for OpenAI provider)
- `OPENAI_MODEL`: OpenAI model to use (default: `gpt-4o-mini`)
- `OLLAMA_MODEL`: Ollama model to use (default: `gemma3:1b`)
- `OLLAMA_HOST`: Ollama server host (default: `localhost`)
- `OLLAMA_PORT`: Ollama server port (default: `11434`)

## Test Data Management

Test data is stored in `/data/` as **JSON** files — each file is a single document with shared metadata (`count`, `attribute`, `options`) plus a list of individual test cases (`images`). **Both OpenAI and Ollama evaluators load the same file** to enable fair model comparison.

### Data Files

- `data/car_color_samples.json` — test cases for car color classification used by both Ollama and OpenAI evaluators (for direct comparison)
- `data/shoe_color_samples.json` — test cases for shoe color classification

### File Structure

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

data = DataLoader.load_json("data/car_color_samples.json")
data["images"].append({"url": "https://example.com/new.jpg", "expected": "red"})

with open("data/car_color_samples.json", "w") as f:
    json.dump(data, f, indent=2)
```

### Using the DataLoader

```python
from src.llm_eval.data import DataLoader

data = DataLoader.load_json("data/car_color_samples.json")
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

### Adding New LLM Providers

1. Create a new adapter in `src/llm_eval/adapters/`:
   ```python
   from .base import LLMAdapter

   class AnthropicAdapter(LLMAdapter):
       def __init__(self, model: str, api_key: str):
           # Initialize client
           pass
       
       def ask(self, question: str) -> str:
           # Implement API call
           pass
       
       @property
       def model_name(self) -> str:
           return self._model
   ```

2. Update `src/llm_eval/adapters/__init__.py` to export it

3. Update `examples/unified_eval.py` to handle the new provider

### Adding New Metrics

Extend `src/llm_eval/core/evaluator.py` with new evaluation functions.

### Adding New Templates

Add new JSON files to `src/llm_eval/templates/` and update the evaluator to support them.

## Testing

Run the full test suite:
```bash
python -m pytest tests/ -v
```

Tests cover the evaluator logic (question generation, accuracy calculation) and data loading (JSON read). No API keys required — all tests use mocks and temporary files.

## Troubleshooting

If you get a `ConnectionError` when running the Ollama evaluator:

1. **Start the Ollama server** (if not already running):
   ```sh
   ollama serve &
   ```
2. **Pull the model** (if not downloaded):
   ```sh
   ollama pull gemma3:1b
   ```
3. **Check server status**:
   ```sh
   ps aux | grep ollama
   ```

The devcontainer should start the server automatically, but you may need to restart it manually.

For demonstration: just share the repo link — anyone with a GitHub account can open it in Codespaces and run it.
