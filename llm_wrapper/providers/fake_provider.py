# ---------------------------------------------------------------------------
# llm_wrapper/providers/fake_provider.py
# ---------------------------------------------------------------------------
# Fake implementation of BaseProvider.
#
# Purpose:
#   - Unit tests
#   - Dry pipeline runs
#   - Debugging state transitions without paying for LLM calls
#
# It implements the same interface as AnthropicProvider/OpenAIProvider/OllamaProvider.
# ---------------------------------------------------------------------------

from __future__ import annotations

from collections import deque
from typing import Any

from llm_wrapper.providers.base_provider import BaseProvider


class FakeProvider(BaseProvider):
    fast_model = "fake-fast"
    smart_model = "fake-smart"

    def __init__(self):
        self.text_responses: deque[str] = deque()
        self.tool_responses: deque[dict[str, Any]] = deque()
        self.calls: list[dict[str, Any]] = []

    def push_text(self, response: str) -> None:
        self.text_responses.append(response)

    def push_tool(self, response: dict[str, Any]) -> None:
        self.tool_responses.append(response)

    def call_text(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        system: str = "",
        images: list[bytes] | None = None,
        effort: str | None = None,
    ) -> str:
        self.calls.append({
            "type": "text",
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "images": self._check_images(images),
            "effort": effort,
        })
        self.last_usage = self._fake_usage(prompt)

        if self.text_responses:
            return self.text_responses.popleft()

        return (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        )

    def call_tool(
        self,
        prompt: str,
        tool_schema: dict,
        tool_name: str,
        model: str,
        max_tokens: int = 4096,
        system: str = "",
        images: list[bytes] | None = None,
        effort: str | None = None,
    ) -> dict:
        self.calls.append({
            "type": "tool",
            "prompt": prompt,
            "tool_schema": tool_schema,
            "tool_name": tool_name,
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "images": self._check_images(images),
            "effort": effort,
        })
        self.last_usage = self._fake_usage(prompt)

        if self.tool_responses:
            return self.tool_responses.popleft()

        return self._default_tool_response(tool_name)

    @staticmethod
    def _fake_usage(prompt: str) -> dict:
        """Usage in the Anthropic provider's format: the prompt's word count as input tokens."""
        return {"input_tokens": len(prompt.split()), "output_tokens": 0,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    @staticmethod
    def _default_tool_response(tool_name: str) -> dict:
        if tool_name == "extract_claims":
            return {
                "claims": [
                    {
                        "text": "lorem ipsum claim requiring literature support",
                        "source_quote": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                        "depth": "shallow",
                    }
                ]
            }

        if tool_name == "classify_file":
            return {
                "label": "other",
                "reason": "Fake provider default classification.",
            }

        if tool_name == "route_compile_error":
            return {
                "needs_repair": False,
                "reason": "Fake provider default compile routing.",
            }

        # Generic fallback for unknown tools.
        # Individual tests should push_tool(...) when exact schema matters.
        return {}