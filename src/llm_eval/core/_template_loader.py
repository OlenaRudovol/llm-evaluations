"""Shared lazy-loading/caching helper for the JSON prompt templates in `templates/`."""

import json
from pathlib import Path
from typing import Dict

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

_cache: Dict[Path, str] = {}


def load_template(config_file: Path, key: str) -> str:
    """Load and cache a named template string from a JSON config file."""
    if config_file in _cache:
        return _cache[config_file]

    try:
        with open(config_file, "r") as f:
            config = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Template config not found at {config_file}. "
            f"Please ensure {config_file.name} exists in the templates directory."
        ) from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_file.name}: {e}") from e

    template = config.get(key, "")
    if not template:
        raise ValueError(f"{key!r} key not found in JSON config")

    _cache[config_file] = template
    return template
