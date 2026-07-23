import json
from pathlib import Path
from typing import Dict


def _check_file_exists(filepath: str) -> Path:
    """Helper: Check if file exists and return Path object.

    Args:
        filepath: Path to file to check

    Returns:
        Path object if file exists

    Raises:
        FileNotFoundError: If file does not exist
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    return path


class DataLoader:
    """Load test data from JSON files."""

    @staticmethod
    def load_json(filepath: str) -> Dict:
        """Load test data from a JSON file."""
        path = _check_file_exists(filepath)
        with open(path, 'r') as f:
            return json.load(f)
