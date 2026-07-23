import json
import pytest
from llm_eval.data.loader import load_json


@pytest.fixture
def sample_records():
    return [
        {"text": "Great value but it broke after a week.", "expected": ["price", "quality"]},
        {"text": "Arrived late but well packaged.", "expected": ["shipping", "packaging"]},
    ]


@pytest.fixture
def json_file(tmp_path, sample_records):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"items": sample_records}))
    return str(path)


class TestLoadJson:
    def test_loads_valid_json(self, json_file):
        data = load_json(json_file)
        assert "items" in data
        assert len(data["items"]) == 2

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_json("/nonexistent/file.json")
