# LLM Evaluation Framework

An open-source framework for evaluating Large Language Models (LLMs) with extensible adapters, templates, datasets, and metrics. Perfect for comparing model performance across different providers and tasks.

## Quick Start

1. **Install**: `pip install -e .`
2. **Configure**: `cp .env.example .env` and fill in your API keys.
3. **Run**:
   ```bash
   python -m examples.unified_eval                     # Ollama (default)
   LLM_PROVIDER=openai python -m examples.unified_eval  # OpenAI
   LLM_PROVIDER=groq python -m examples.unified_eval    # Groq (free tier at console.groq.com)
   ```
4. Running locally? Install Ollama (ollama.com) and pull a model: `ollama pull gemma3:1b`

## Project Structure

```
llm-eval/
├── src/llm_eval/          # Core package
│   ├── adapters/          # LLM provider adapters (Ollama, OpenAI, Groq, etc.)
│   ├── core/              # Evaluation engine and configuration
│   ├── data/              # Data loading utilities
│   └── templates/         # Question/prompt templates
├── examples/              # Demo scripts and usage examples
├── data/                  # Datasets
├── scripts/               # One-off data tooling (not part of the eval pipeline)
└── tests/                 # Unit tests
```

## Configuration

All settings are environment variables (see `.env.example`):

- `LLM_PROVIDER` — `ollama` (default), `openai`, or `groq`
- `OLLAMA_MODEL` (default `gemma3:1b`), `OPENAI_MODEL` (default `gpt-4o-mini`), `GROQ_MODEL` (default `llama-3.1-8b-instant`)
- `OPENAI_API_KEY` / `GROQ_API_KEY` — required for that provider (Groq: free tier, no card, at [console.groq.com](https://console.groq.com))
- `USE_LLM_JUDGE` — `true` to additionally run LLM-as-judge (default `false`)
- `JUDGE_MODEL` — stronger model to judge with, same provider (default: unset → judges with the model being tested)
- `EVAL_DATA_FILE` — which `data/*.json` file to evaluate (default `data/review_aspects_samples.json`)

## Evaluation Methods

Every run scores answers with **substring matching**: an aspect counts as predicted if it appears as a case-insensitive substring of the model's answer. Fast and free, but brittle — "cost" instead of "price" is marked wrong. Reports four views: **Average F1** (macro), **Exact match rate**, **Micro P/R/F1**, and **Per-label P/R/F1** (`evaluator.per_label_prf1`, with `support`).

`USE_LLM_JUDGE=true` additionally asks a second LLM to judge correctness in natural language, tolerating paraphrases substring matching would reject. `JUDGE_MODEL` overrides which model does the judging.

Running `python -m examples.unified_eval` prints a full explanation of what each number means and why it's reported, right alongside the results — no need to look it up here.

## Test Data

Files in `/data/` are JSON: `{version, updated, attribute, options, reviews: [{text, expected}]}`. Every provider evaluates against the same file.

- **`review_aspects_samples.json`** — 16 hand-written reviews covering single-label baselines, keyword-coincidence distractors, and indirect phrasing.
- **`absa_restaurant_samples.json`** — 200 real restaurant reviews from a public Hugging Face ABSA dataset (see below). Scores much lower than the curated set — a small hand-written dataset can quietly become "too easy."

```json
{
  "version": "1.0",
  "updated": "2026-07-24",
  "attribute": "aspects",
  "options": ["price", "quality", "shipping", "customer service", "packaging"],
  "reviews": [
    { "text": "Great value but it broke after a week.", "expected": ["price", "quality"] }
  ]
}
```

**Adding samples** — edit the JSON directly, or via `load_json`:
```python
import json
from llm_eval.data import load_json

data = load_json("data/review_aspects_samples.json")
data["reviews"].append({"text": "Support was rude.", "expected": ["customer service"]})
json.dump(data, open("data/review_aspects_samples.json", "w"), indent=2)
```

**Best practices**: bump `version`/`updated` on any change — results from different versions aren't comparable; generate data via script rather than hand-editing where possible; keep splits tagged; don't commit sensitive data. `tests/test_data_integrity.py` automatically checks every `data/*.json` file for label drift, duplicates, and version metadata.

## Building a Dataset from Real Data

`scripts/explore_absa_dataset.py` and `scripts/build_absa_sample.py` (`pip install -e ".[analysis]"`) pull a real ABSA dataset from Hugging Face, run hygiene checks (duplicates, label balance, train/test leakage), and sample it into our schema:

```bash
python -m scripts.explore_absa_dataset   # hygiene/analysis report
python -m scripts.build_absa_sample       # (re)writes data/absa_restaurant_samples.json
```

The sample is a reproducible random draw (`seed=42`), not stratified — it keeps the source data's real class imbalance instead of hiding it. To adapt this to another dataset, change `DATASET_ID`, `OPTIONS`, and `to_categories()` in `build_absa_sample.py`.

## Extending the Framework

**New LLM provider** — subclass `LLMAdapter` in `src/llm_eval/adapters/`:
```python
class NewProviderAdapter(LLMAdapter):
    def __init__(self, model: str, api_key: str): ...
    def ask(self, question: str) -> str: ...
    @property
    def model_name(self) -> str: ...
```
Export it from `adapters/__init__.py`, then wire it into `create_adapter()` in `examples/unified_eval.py`.

**New metric** — add a function in `evaluator.py` taking `(predicted, expected)` label-set pairs and returning a score (see `exact_match_eval`, `per_label_prf1`). For judge-style metrics needing their own LLM calls, add a module like `judge.py`.

**New template** — add a JSON file to `src/llm_eval/templates/` and wire it into the evaluator.

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Covers evaluator/scoring logic, judge verdict parsing, data loading, all three adapters (mocked, no network), and dataset integrity. No API keys required.

## Troubleshooting

`ConnectionError` from the Ollama evaluator:
1. `ollama serve &` — start the server
2. `ollama pull gemma3:1b` — pull the model
3. `ps aux | grep ollama` — check it's running

The devcontainer starts Ollama automatically; restart manually if needed. For demos: share the repo link — anyone can open it in Codespaces and run it.
