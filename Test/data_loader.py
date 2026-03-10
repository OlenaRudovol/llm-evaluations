import json
from pathlib import Path
from typing import List, Dict, Generator


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


def _parse_json_lines(file_obj) -> Generator[Dict, None, None]:
    """Helper: Parse JSON lines from file object and yield dictionaries.
    
    Yields:
        Dictionary for each non-empty line
    """
    for line in file_obj:
        line = line.strip()
        if line:  # skip empty lines
            yield json.loads(line)


class DataLoader:
    """Load test data from JSONL files."""

    @staticmethod
    def load_jsonl(filepath: str) -> List[Dict]:
        """Load test data from JSONL file (one JSON object per line)."""
        path = _check_file_exists(filepath)
        with open(path, 'r') as f:
            return list(_parse_json_lines(f))

    @staticmethod
    def save_jsonl(samples: List[Dict], filepath: str) -> None:
        """Save samples to JSONL file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

    @staticmethod
    def stream_jsonl(filepath: str) -> Generator[Dict, None, None]:
        """Stream test data from JSONL file instead of loading all into memory.
        
        This is memory-efficient for large datasets.
        Yields one JSON object per line.
        
        Args:
            filepath: Path to JSONL file
            
        Yields:
            Dictionary for each JSON line in the file
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        path = _check_file_exists(filepath)
        with open(path, 'r') as f:
            yield from _parse_json_lines(f)
