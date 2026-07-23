import json
import pytest
from src.llm_eval.data.loader import DataLoader


@pytest.fixture
def sample_records():
    return [
        {"count": 1, "attribute": "car colour", "options": ["red", "blue"]},
        {"count": 2, "attribute": "shoe colour", "options": ["black", "white"]},
    ]


@pytest.fixture
def json_file(tmp_path, sample_records):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"items": sample_records}))
    return str(path)


class TestLoadJson:
    def test_loads_valid_json(self, json_file):
        data = DataLoader.load_json(json_file)
        assert "items" in data
        assert len(data["items"]) == 2

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            DataLoader.load_json("/nonexistent/file.json")
