from unittest.mock import MagicMock

import pytest

from llm_eval.adapters.openai import OpenAIAdapter

MODEL = "gpt-4o-mini"


class TestOpenAIAdapter:
    @pytest.fixture
    def adapter(self):
        return OpenAIAdapter(model=MODEL, api_key="sk-test")

    @pytest.mark.parametrize("api_key", ["", None])
    def test_missing_api_key_raises(self, api_key):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIAdapter(model=MODEL, api_key=api_key)

    def test_model_name(self, adapter):
        assert adapter.model_name == MODEL

    def test_ask_returns_stripped_content(self, adapter):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  Price, Quality  \n"
        adapter.client.chat.completions.create = MagicMock(return_value=mock_response)

        answer = adapter.ask("What aspects?")

        assert answer == "Price, Quality"
        adapter.client.chat.completions.create.assert_called_once_with(
            model=MODEL,
            messages=[{"role": "user", "content": "What aspects?"}],
            temperature=0.0,
        )

    def test_ask_wraps_malformed_response(self, adapter):
        mock_response = MagicMock()
        mock_response.choices = []  # choices[0] -> IndexError
        adapter.client.chat.completions.create = MagicMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Invalid response format"):
            adapter.ask("What aspects?")

    def test_ask_wraps_api_errors(self, adapter):
        adapter.client.chat.completions.create = MagicMock(side_effect=TimeoutError("timed out"))

        with pytest.raises(RuntimeError, match="OpenAI API error"):
            adapter.ask("What aspects?")

    def test_ask_preserves_exception_chain(self, adapter):
        original = TimeoutError("timed out")
        adapter.client.chat.completions.create = MagicMock(side_effect=original)

        with pytest.raises(RuntimeError) as exc_info:
            adapter.ask("What aspects?")

        assert exc_info.value.__cause__ is original
