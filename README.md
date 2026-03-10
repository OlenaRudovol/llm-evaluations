# LLM Evaluation Demo

This repo is for simple

## How to run
1. Open in GitHub Codespaces (click the green "Code" > "Codespaces" > "Create codespace on main").
2. **Set up environment variables**: Copy `.env.example` to `.env` and fill in your API keys:
   ```sh
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```
   The scripts will automatically load variables from `.env` if it exists.
3. **Install dependencies** (if not already done by devcontainer):
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the evaluator** with your preferred LLM provider:
   ```sh
   # Using default provider (Ollama)
   python -m Test.eval
   
   # Or use OpenAI
   LLM_PROVIDER=openai python -m Test.eval
   ```
5. If running locally, install Ollama (ollama.com), clone, `pip install -r requirements.txt`, `ollama pull gemma3:1b`, then invoke the same commands above.

## Configuration

The evaluator uses environment variables for configuration. See `.env.example` for all available options:

- `LLM_PROVIDER`: Which provider to use (default: `ollama`, options: `ollama`, `openai`)
- `OPENAI_API_KEY`: Your OpenAI API key (required for OpenAI provider)
- `OPENAI_MODEL`: OpenAI model to use (default: `gpt-4o-mini`)
- `OLLAMA_MODEL`: Ollama model to use (default: `gemma3:1b`)
- `OLLAMA_HOST`: Ollama server host (default: `localhost`)
- `OLLAMA_PORT`: Ollama server port (default: `11434`)

## Test Data Management

Test data is stored in `/data/` as **JSONL** (JSON Lines) files — one JSON object per line. **Both OpenAI and Ollama evaluators use the same dataset** to enable fair model comparison.

This format:
- Scales to millions of samples without memory issues (stream-friendly)
- Is human-readable and easy to version control
- Supports incremental loading and filtering

### Data Files

- `data/car_color_samples.jsonl` — test cases for car color classification used by both Ollama and OpenAI evaluators (for direct comparison)

### Adding More Samples

**Option 1: Edit the JSONL file directly**
```jsonl
{"question": "New question?", "expected": "expected answer"}
{"question": "Another question?", "expected": "answer"}
```

**Option 2: Generate samples programmatically**
```python
from Test.data_loader import DataLoader

samples = [
    {"question": "Q1?", "expected": "A1"},
    {"question": "Q2?", "expected": "A2"},
    # ... up to 3000+ samples
]

DataLoader.save_jsonl(samples, "data/my_samples.jsonl")
```

### Using the DataLoader

```python
from Test.data_loader import DataLoader

# Load a JSONL file
samples = DataLoader.load_jsonl("data/my_samples.jsonl")

# Stream large datasets (memory efficient)
for sample in DataLoader.stream_jsonl("data/my_samples.jsonl"):
    print(sample)

# Save samples to JSONL
DataLoader.save_jsonl(samples, "data/output.jsonl")
```

### Best Practices for Large Datasets (100-3000+ samples)

1. **Use JSONL format** — streams efficiently, no memory bloat
2. **Version your data** — tag datasets with dates or version numbers
3. **Make data regenerable** — script your data generation (not hand-curated)
4. **Organize by split** — keep train/test/val separate or tagged
5. **Document schema** — add comments in your data files about expected fields
6. **Consider data privacy** — don't commit sensitive test data to git

### Supported Formats

- **JSONL** (recommended) — `/data/samples.jsonl` (memory-efficient for large datasets)

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
