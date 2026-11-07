# ---------------------------------------------------------------------------
# tests/test_llm.py
# ---------------------------------------------------------------------------
# Unit tests  — always run; use FakeProvider, no API keys needed.
# Integration — live calls per provider; skipped when key is absent.
#               Ollama image tests also need OLLAMA_VISION_MODEL set.
#
# Run all:         pytest tests/test_llm.py -v
# Unit only:       pytest tests/test_llm.py -v -m "not integration"
# One provider:    pytest tests/test_llm.py -v -k "anthropic"
# ---------------------------------------------------------------------------

import base64
import json
import os
import struct
import zlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import llm_wrapper.llm as llm_module
from llm_wrapper.llm import call_llm, call_llm_with_tool
from llm_wrapper.providers.base_provider import BaseProvider
from llm_wrapper.providers.fake_provider import FakeProvider


def _solid_png(rgb: tuple[int, int, int], size: int = 128) -> bytes:
    """Build a solid-color PNG with the stdlib, so tests need no imaging dependency."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    row = b"\x00" + bytes(rgb) * size
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


_RGB = {"red": (255, 0, 0), "blue": (0, 0, 255)}
_RED_PNG = _solid_png(_RGB["red"])
_BLUE_PNG = _solid_png(_RGB["blue"])
_TWO_IMAGES = [_RED_PNG, _BLUE_PNG]
_JPEG_BYTES = b"\xff\xd8\xff\xe0 not a png"


def _b64(image: bytes) -> str:
    return base64.b64encode(image).decode("ascii")


_COLOR_SCHEMA = {
    "name": "report_color",
    "description": "Report the color that fills the image.",
    "input_schema": {
        "type": "object",
        "properties": {
            "color": {"type": "string", "enum": ["red", "green", "blue"]},
        },
        "required": ["color"],
    },
}

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


class TestImageValidation:
    def test_none_means_no_images(self):
        assert BaseProvider._check_images(None) == []

    def test_png_accepted(self):
        assert BaseProvider._check_images(_TWO_IMAGES) == _TWO_IMAGES

    def test_base64_round_trip(self):
        assert BaseProvider._images_to_base64([_RED_PNG]) == [_b64(_RED_PNG)]

    @pytest.mark.parametrize("bad", [
        [_JPEG_BYTES],
        [_RED_PNG, _JPEG_BYTES],
        ["/path/to/image.png"],
        _RED_PNG,  # bare bytes instead of a list
    ])
    def test_non_png_rejected(self, bad):
        with pytest.raises(ValueError, match="not PNG bytes"):
            BaseProvider._check_images(bad)


class TestCallLLMImages:
    def _inject(self, monkeypatch):
        fake = FakeProvider()
        monkeypatch.setattr(llm_module, "_cache", {"fake": fake})
        return fake

    def test_call_llm_forwards_images(self, monkeypatch):
        fake = self._inject(monkeypatch)
        call_llm("p", model="m", provider="fake", images=_TWO_IMAGES)
        assert fake.calls[0]["images"] == _TWO_IMAGES

    def test_call_llm_with_tool_forwards_images(self, monkeypatch):
        fake = self._inject(monkeypatch)
        call_llm_with_tool("p", _SENTIMENT_SCHEMA, "classify_sentiment", model="m", provider="fake", images=_TWO_IMAGES)
        assert fake.calls[0]["images"] == _TWO_IMAGES

    def test_images_default_to_none_sent(self, monkeypatch):
        fake = self._inject(monkeypatch)
        call_llm("p", model="m", provider="fake")
        assert fake.calls[0]["images"] == []

    def test_non_png_raises_through_top_level(self, monkeypatch):
        self._inject(monkeypatch)
        with pytest.raises(ValueError, match="not PNG bytes"):
            call_llm("p", model="m", provider="fake", images=[_JPEG_BYTES])

    def test_log_records_count_not_bytes(self, monkeypatch, tmp_path):
        self._inject(monkeypatch)
        monkeypatch.setattr(llm_module, "LOG_LLM_CALLS", True)
        monkeypatch.setattr(llm_module, "LOGS_DIR", tmp_path)
        call_llm("p", model="m", provider="fake", images=_TWO_IMAGES)
        call_llm("p", model="m", provider="fake")
        lines = (tmp_path / "llm_calls.jsonl").read_text().splitlines()
        assert json.loads(lines[0])["image_count"] == 2
        assert "image_count" not in json.loads(lines[1])
        assert _b64(_RED_PNG) not in lines[0]


def _text_call(provider, images):
    provider.call_text("p", model="m", max_tokens=5, images=images)


def _tool_call(provider, images):
    provider.call_tool("p", _SENTIMENT_SCHEMA, "classify_sentiment", model="m", max_tokens=5, images=images)


_BOTH_CALLS = pytest.mark.parametrize("call", [_text_call, _tool_call], ids=["text", "tool"])


class TestAnthropicRequest:
    def _provider(self):
        from llm_wrapper.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.client = MagicMock()
        provider.client.messages.create.return_value = SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="text", text="ok"),
                SimpleNamespace(type="tool_use", name="classify_sentiment", input={"label": "neutral"}),
            ],
        )
        return provider

    def _content(self, provider):
        return provider.client.messages.create.call_args.kwargs["messages"][0]["content"]

    @_BOTH_CALLS
    def test_images_precede_text(self, call):
        provider = self._provider()
        call(provider, _TWO_IMAGES)
        assert self._content(provider) == [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(_RED_PNG)}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(_BLUE_PNG)}},
            {"type": "text", "text": "p"},
        ]

    @_BOTH_CALLS
    @pytest.mark.parametrize("empty", [None, []])
    def test_no_images_keeps_plain_string(self, call, empty):
        provider = self._provider()
        call(provider, empty)
        assert self._content(provider) == "p"

    @_BOTH_CALLS
    def test_non_png_raises_before_request(self, call):
        provider = self._provider()
        with pytest.raises(ValueError):
            call(provider, [_JPEG_BYTES])
        provider.client.messages.create.assert_not_called()


class TestOpenAIRequest:
    def _provider(self):
        from llm_wrapper.providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.client = MagicMock()
        tool_call = SimpleNamespace(function=SimpleNamespace(arguments='{"label": "neutral"}'))
        provider.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="ok", tool_calls=[tool_call]),
            )],
        )
        return provider

    def _content(self, provider):
        return provider.client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]

    @_BOTH_CALLS
    def test_images_precede_text(self, call):
        provider = self._provider()
        call(provider, _TWO_IMAGES)
        assert self._content(provider) == [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(_RED_PNG)}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(_BLUE_PNG)}"}},
            {"type": "text", "text": "p"},
        ]

    @_BOTH_CALLS
    @pytest.mark.parametrize("empty", [None, []])
    def test_no_images_keeps_plain_string(self, call, empty):
        provider = self._provider()
        call(provider, empty)
        assert self._content(provider) == "p"

    @_BOTH_CALLS
    def test_non_png_raises_before_request(self, call):
        provider = self._provider()
        with pytest.raises(ValueError):
            call(provider, [_JPEG_BYTES])
        provider.client.chat.completions.create.assert_not_called()


class TestGeminiRequest:
    def _provider(self):
        types = pytest.importorskip("google.genai.types")
        from llm_wrapper.providers.gemini_provider import GeminiProvider
        provider = GeminiProvider.__new__(GeminiProvider)
        provider._types = types
        provider.client = MagicMock()
        function_call = SimpleNamespace(name="classify_sentiment", args={"label": "neutral"})
        provider.client.models.generate_content.return_value = SimpleNamespace(
            text="ok",
            candidates=[SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(parts=[SimpleNamespace(function_call=function_call)]),
            )],
        )
        return provider

    def _contents(self, provider):
        return provider.client.models.generate_content.call_args.kwargs["contents"]

    @_BOTH_CALLS
    def test_images_precede_text(self, call):
        provider = self._provider()
        call(provider, _TWO_IMAGES)
        first, second, text = self._contents(provider)
        assert (first.inline_data.data, first.inline_data.mime_type) == (_RED_PNG, "image/png")
        assert (second.inline_data.data, second.inline_data.mime_type) == (_BLUE_PNG, "image/png")
        assert text == "p"

    @_BOTH_CALLS
    @pytest.mark.parametrize("empty", [None, []])
    def test_no_images_keeps_plain_string(self, call, empty):
        provider = self._provider()
        call(provider, empty)
        assert self._contents(provider) == "p"

    @_BOTH_CALLS
    def test_non_png_raises_before_request(self, call):
        provider = self._provider()
        with pytest.raises(ValueError):
            call(provider, [_JPEG_BYTES])
        provider.client.models.generate_content.assert_not_called()


class _FakeOllamaServer:
    """Stands in for requests.post: answers /api/show and scripted /api/chat replies."""

    def __init__(self, capabilities=("completion", "vision", "tools"), chat_replies=(), show_error=None):
        self.capabilities = list(capabilities)
        self.chat_replies = list(chat_replies)
        self.show_error = show_error
        self.show_calls = 0
        self.chat_payloads = []

    def post(self, url, json, timeout):
        if url.endswith("/api/show"):
            self.show_calls += 1
            if self.show_error:
                raise self.show_error
            body = {"capabilities": self.capabilities}
        else:
            self.chat_payloads.append(json)
            reply = self.chat_replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            body = reply
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)


def _ollama_text(content: str) -> dict:
    return {"message": {"content": content}, "done_reason": "stop"}


_OLLAMA_NATIVE_TOOL_CALL = {
    "message": {"tool_calls": [{"function": {"arguments": {"label": "neutral"}}}]},
    "done_reason": "stop",
}


class TestOllamaRequest:
    def _setup(self, monkeypatch, **server_kwargs):
        from llm_wrapper.providers.ollama_provider import OllamaProvider
        server = _FakeOllamaServer(**server_kwargs)
        monkeypatch.setattr("llm_wrapper.providers.ollama_provider.requests.post", server.post)
        return OllamaProvider(), server

    def _tool(self, provider, images):
        return provider.call_tool("p", _SENTIMENT_SCHEMA, "classify_sentiment", model="m", max_tokens=5, images=images)

    def test_no_images_sends_plain_message_and_skips_vision_check(self, monkeypatch):
        provider, server = self._setup(monkeypatch, chat_replies=[_ollama_text("ok")])
        provider.call_text("p", model="m", max_tokens=5)
        assert server.chat_payloads[0]["messages"] == [{"role": "user", "content": "p"}]
        assert server.show_calls == 0

    def test_text_call_sends_images_on_user_message_only(self, monkeypatch):
        provider, server = self._setup(monkeypatch, chat_replies=[_ollama_text("ok")])
        provider.call_text("p", model="m", max_tokens=5, system="s", images=_TWO_IMAGES)
        assert server.chat_payloads[0]["messages"] == [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "p", "images": [_b64(_RED_PNG), _b64(_BLUE_PNG)]},
        ]

    def test_native_tool_call_sends_images(self, monkeypatch):
        provider, server = self._setup(monkeypatch, chat_replies=[_OLLAMA_NATIVE_TOOL_CALL])
        assert self._tool(provider, [_RED_PNG]) == {"label": "neutral"}
        assert "tools" in server.chat_payloads[0]
        assert server.chat_payloads[0]["messages"][-1]["images"] == [_b64(_RED_PNG)]

    def test_model_without_tools_sends_images_via_json_prompt(self, monkeypatch):
        provider, server = self._setup(
            monkeypatch,
            capabilities=("completion", "vision"),
            chat_replies=[_ollama_text('{"label": "neutral"}')],
        )
        assert self._tool(provider, [_RED_PNG]) == {"label": "neutral"}
        assert "tools" not in server.chat_payloads[0]
        assert server.chat_payloads[0]["messages"][-1]["images"] == [_b64(_RED_PNG)]

    def test_native_failure_fallback_keeps_images(self, monkeypatch):
        provider, server = self._setup(
            monkeypatch,
            chat_replies=[RuntimeError("HTTP 500"), _ollama_text('{"label": "neutral"}')],
        )
        assert self._tool(provider, [_RED_PNG]) == {"label": "neutral"}
        assert len(server.chat_payloads) == 2
        assert all(p["messages"][-1]["images"] == [_b64(_RED_PNG)] for p in server.chat_payloads)

    def test_json_repair_keeps_images(self, monkeypatch):
        provider, server = self._setup(
            monkeypatch,
            capabilities=("completion", "vision"),
            chat_replies=[_ollama_text("not json"), _ollama_text('{"label": "neutral"}')],
        )
        assert self._tool(provider, [_RED_PNG]) == {"label": "neutral"}
        assert len(server.chat_payloads) == 2
        assert all(p["messages"][-1]["images"] == [_b64(_RED_PNG)] for p in server.chat_payloads)

    def test_text_only_model_raises_before_any_chat_request(self, monkeypatch):
        provider, server = self._setup(monkeypatch, capabilities=("completion", "tools"))
        with pytest.raises(RuntimeError, match="does not support images"):
            provider.call_text("p", model="m", max_tokens=5, images=[_RED_PNG])
        with pytest.raises(RuntimeError, match="does not support images"):
            self._tool(provider, [_RED_PNG])  # must not be swallowed by the native-path except
        assert server.chat_payloads == []

    def test_failed_vision_check_raises_then_recovers(self, monkeypatch):
        provider, server = self._setup(
            monkeypatch,
            chat_replies=[_ollama_text("ok")],
            show_error=ConnectionError("hiccup"),
        )
        with pytest.raises(RuntimeError, match="Could not confirm"):
            provider.call_text("p", model="m", max_tokens=5, images=[_RED_PNG])
        assert server.chat_payloads == []

        server.show_error = None
        assert provider.call_text("p", model="m", max_tokens=5, images=[_RED_PNG]) == "ok"

    def test_confirmed_vision_is_cached(self, monkeypatch):
        provider, server = self._setup(monkeypatch, chat_replies=[_ollama_text("a"), _ollama_text("b")])
        provider.call_text("p", model="m", max_tokens=5, images=[_RED_PNG])
        provider.call_text("p", model="m", max_tokens=5, images=[_RED_PNG])
        assert server.show_calls == 1

    def test_non_png_raises_before_any_request(self, monkeypatch):
        provider, server = self._setup(monkeypatch)
        with pytest.raises(ValueError):
            provider.call_text("p", model="m", max_tokens=5, images=[_JPEG_BYTES])
        assert server.show_calls == 0 and server.chat_payloads == []


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class _LiveImageTests:
    """Red and blue images must get different answers, which proves the image arrived."""

    IMAGE_MODEL = None

    def _image_model(self):
        if not self.IMAGE_MODEL:
            pytest.skip("No vision model configured for this provider")
        return self.IMAGE_MODEL

    @pytest.mark.parametrize("color", sorted(_RGB))
    def test_call_text_with_image(self, color):
        result = self.provider.call_text(
            "What color fills this image? Reply with one word: red, green, or blue.",
            model=self._image_model(),
            max_tokens=300,
            images=[_solid_png(_RGB[color])],
        )
        assert color in result.lower()

    @pytest.mark.parametrize("color", sorted(_RGB))
    def test_call_tool_with_image(self, color):
        result = self.provider.call_tool(
            "Report the color that fills this image.",
            _COLOR_SCHEMA,
            "report_color",
            model=self._image_model(),
            max_tokens=300,
            images=[_solid_png(_RGB[color])],
        )
        assert result.get("color") == color


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestAnthropicProvider(_LiveImageTests):
    MODEL = "claude-haiku-4-5"
    IMAGE_MODEL = MODEL

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
class TestOpenAIProvider(_LiveImageTests):
    MODEL = "gpt-4o-mini"
    IMAGE_MODEL = MODEL

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
class TestGeminiProvider(_LiveImageTests):
    MODEL = "gemini-3.5-flash"
    IMAGE_MODEL = MODEL

    def setup_method(self):
        from llm_wrapper.providers.gemini_provider import GeminiProvider
        self.provider = GeminiProvider()

    # Budgets are higher than other providers': thinking tokens count toward max_tokens here.
    def test_call_text(self):
        result = self.provider.call_text("Reply with just the word hello.", model=self.MODEL, max_tokens=300)
        assert isinstance(result, str) and len(result) > 0

    def test_call_tool(self):
        result = self.provider.call_tool("I love this product!", _SENTIMENT_SCHEMA, "classify_sentiment", model=self.MODEL, max_tokens=300)
        assert result.get("label") in ("positive", "negative", "neutral")


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OLLAMA_BASE_URL", ""), reason="OLLAMA_BASE_URL not set")
class TestOllamaProvider(_LiveImageTests):
    MODEL = os.getenv("MODEL", "llama3")
    IMAGE_MODEL = os.getenv("OLLAMA_VISION_MODEL")

    def setup_method(self):
        from llm_wrapper.providers.ollama_provider import OllamaProvider
        self.provider = OllamaProvider()

    def test_call_text(self):
        result = self.provider.call_text("Reply with just the word hello.", model=self.MODEL, max_tokens=20)
        assert isinstance(result, str) and len(result) > 0

    def test_call_tool(self):
        result = self.provider.call_tool("I love this product!", _SENTIMENT_SCHEMA, "classify_sentiment", model=self.MODEL, max_tokens=64)
        assert result.get("label") in ("positive", "negative", "neutral")
