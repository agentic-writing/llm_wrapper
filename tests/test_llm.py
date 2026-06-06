# ---------------------------------------------------------------------------
# tests/test_llm.py
# ---------------------------------------------------------------------------
# Unit tests  — always run; use FakeProvider, no API keys needed.
# Integration — one live call per provider; skipped when key is absent.
#
# Run all:         pytest tests/test_llm.py -v
# Unit only:       pytest tests/test_llm.py -v -m "not integration"
# One provider:    pytest tests/test_llm.py -v -k "anthropic"
# ---------------------------------------------------------------------------

import os

import pytest

import llm_wrapper.llm as llm_module
from llm_wrapper.llm import call_llm, call_llm_with_tool
from llm_wrapper.providers.fake_provider import FakeProvider

_SENTIMENT_SCHEMA = {
    "name": "classify_sentiment",
    "description": "Classify the sentiment of the input text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
                "description": "Sentiment label.",
            }
        },
        "required": ["label"],
    },
}

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestFakeProvider:
    def setup_method(self):
        self.fake = FakeProvider()

    def test_text_returns_pushed_response(self):
        self.fake.push_text("hello world")
        assert self.fake.call_text("any prompt", model="m", max_tokens=10) == "hello world"

    def test_text_default_when_nothing_pushed(self):
        result = self.fake.call_text("any prompt", model="m", max_tokens=10)
        assert isinstance(result, str) and len(result) > 0

    def test_tool_returns_pushed_response(self):
        self.fake.push_tool({"label": "positive"})
        result = self.fake.call_tool("great!", _SENTIMENT_SCHEMA, "classify_sentiment", model="m", max_tokens=10)
        assert result == {"label": "positive"}

    def test_tool_default_is_dict(self):
        result = self.fake.call_tool("anything", _SENTIMENT_SCHEMA, "classify_sentiment", model="m", max_tokens=10)
        assert isinstance(result, dict)

    def test_calls_log_populated(self):
        self.fake.push_text("ok")
        self.fake.call_text("probe", model="m", max_tokens=5)
        assert self.fake.calls[0]["prompt"] == "probe"
        assert self.fake.calls[0]["type"] == "text"

    def test_multiple_pushes_in_order(self):
        self.fake.push_text("first")
        self.fake.push_text("second")
        assert self.fake.call_text("x", model="m", max_tokens=5) == "first"
        assert self.fake.call_text("x", model="m", max_tokens=5) == "second"


class TestProviderInference:
    def test_claude_infers_anthropic(self):
        from llm_wrapper.llm import _infer_provider
        assert _infer_provider("claude-haiku-4-5") == "anthropic"

    def test_gpt_infers_openai(self):
        from llm_wrapper.llm import _infer_provider
        assert _infer_provider("gpt-4o-mini") == "openai"

    def test_gemini_infers_gemini(self):
        from llm_wrapper.llm import _infer_provider
        assert _infer_provider("gemini-2.0-flash") == "gemini"

    def test_unknown_returns_none(self):
        from llm_wrapper.llm import _infer_provider
        assert _infer_provider("llama3") is None

    def test_explicit_provider_wins(self):
        from llm_wrapper.llm import _resolve_provider
        assert _resolve_provider("ollama", "claude-haiku-4-5") == "ollama"


class TestCallLLM:
    def _inject(self, monkeypatch, fake):
        monkeypatch.setattr(llm_module, "_cache", {"fake": fake})

    def test_call_llm_returns_text(self, monkeypatch):
        fake = FakeProvider()
        fake.push_text("routed correctly")
        self._inject(monkeypatch, fake)
        assert call_llm("hello", model="m", provider="fake") == "routed correctly"

    def test_call_llm_with_tool_returns_dict(self, monkeypatch):
        fake = FakeProvider()
        fake.push_tool({"label": "negative"})
        self._inject(monkeypatch, fake)
        result = call_llm_with_tool("Terrible.", _SENTIMENT_SCHEMA, "classify_sentiment", model="m", provider="fake")
        assert result == {"label": "negative"}

    def test_prompt_passed_through(self, monkeypatch):
        fake = FakeProvider()
        fake.push_text("ok")
        self._inject(monkeypatch, fake)
        call_llm("my exact prompt", model="m", provider="fake")
        assert fake.calls[0]["prompt"] == "my exact prompt"

    def test_schema_passed_through(self, monkeypatch):
        fake = FakeProvider()
        fake.push_tool({"label": "neutral"})
        self._inject(monkeypatch, fake)
        call_llm_with_tool("test", _SENTIMENT_SCHEMA, "classify_sentiment", model="m", provider="fake")
        assert fake.calls[0]["tool_schema"] == _SENTIMENT_SCHEMA

    def test_two_providers_cached_separately(self, monkeypatch):
        fake1, fake2 = FakeProvider(), FakeProvider()
        fake1.push_text("from fake1")
        fake2.push_text("from fake2")
        monkeypatch.setattr(llm_module, "_cache", {"prov-a": fake1, "prov-b": fake2})
        assert call_llm("x", model="m", provider="prov-a") == "from fake1"
        assert call_llm("x", model="m", provider="prov-b") == "from fake2"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestAnthropicProvider:
    MODEL = "claude-haiku-4-5"

    def setup_method(self):
        from llm_wrapper.providers.anthropic_provider import AnthropicProvider
        self.provider = AnthropicProvider()

    def test_call_text(self):
        result = self.provider.call_text("Reply with just the word hello.", model=self.MODEL, max_tokens=20)
        assert isinstance(result, str) and len(result) > 0

    def test_call_tool(self):
        result = self.provider.call_tool("I love this product!", _SENTIMENT_SCHEMA, "classify_sentiment", model=self.MODEL, max_tokens=64)
        assert result.get("label") in ("positive", "negative", "neutral")


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestOpenAIProvider:
    MODEL = "gpt-4o-mini"

    def setup_method(self):
        from llm_wrapper.providers.openai_provider import OpenAIProvider
        self.provider = OpenAIProvider()

    def test_call_text(self):
        result = self.provider.call_text("Reply with just the word hello.", model=self.MODEL, max_tokens=20)
        assert isinstance(result, str) and len(result) > 0

    def test_call_tool(self):
        result = self.provider.call_tool("I love this product!", _SENTIMENT_SCHEMA, "classify_sentiment", model=self.MODEL, max_tokens=64)
        assert result.get("label") in ("positive", "negative", "neutral")


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason="GEMINI_API_KEY not set",
)
class TestGeminiProvider:
    MODEL = "gemini-2.0-flash"

    def setup_method(self):
        from llm_wrapper.providers.gemini_provider import GeminiProvider
        self.provider = GeminiProvider()

    def test_call_text(self):
        result = self.provider.call_text("Reply with just the word hello.", model=self.MODEL, max_tokens=20)
        assert isinstance(result, str) and len(result) > 0

    def test_call_tool(self):
        result = self.provider.call_tool("I love this product!", _SENTIMENT_SCHEMA, "classify_sentiment", model=self.MODEL, max_tokens=64)
        assert result.get("label") in ("positive", "negative", "neutral")


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OLLAMA_BASE_URL", ""), reason="OLLAMA_BASE_URL not set")
class TestOllamaProvider:
    MODEL = os.getenv("MODEL", "llama3")

    def setup_method(self):
        from llm_wrapper.providers.ollama_provider import OllamaProvider
        self.provider = OllamaProvider()

    def test_call_text(self):
        result = self.provider.call_text("Reply with just the word hello.", model=self.MODEL, max_tokens=20)
        assert isinstance(result, str) and len(result) > 0

    def test_call_tool(self):
        result = self.provider.call_tool("I love this product!", _SENTIMENT_SCHEMA, "classify_sentiment", model=self.MODEL, max_tokens=64)
        assert result.get("label") in ("positive", "negative", "neutral")
