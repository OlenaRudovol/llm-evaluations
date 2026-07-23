import json
from typing import Dict


def load_json(filepath: str) -> Dict:
    """Load test data from a JSON file.

    Raises:
        FileNotFoundError: If filepath does not exist (raised by `open`).
    """
    with open(filepath, "r") as f:
        return json.load(f)
