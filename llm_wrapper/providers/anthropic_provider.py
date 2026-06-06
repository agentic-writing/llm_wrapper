# ---------------------------------------------------------------------------
# llm_wrapper/providers/anthropic_provider.py
# ---------------------------------------------------------------------------
# Anthropic implementation of BaseProvider. Wraps the Anthropic Python SDK
# and hides all API-specific details (response parsing, tool extraction,
# system prompt format) from the rest of the framework.
# ---------------------------------------------------------------------------

import os

from anthropic import Anthropic

from llm_wrapper.providers.base_provider import BaseProvider


class AnthropicProvider(BaseProvider):

    def __init__(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY missing")
        self.client = Anthropic()

    def call_text(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        system: str = "",
    ) -> str:
        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)

        return "\n".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()

    def call_tool(
        self,
        prompt: str,
        tool_schema: dict,
        tool_name: str,
        model: str,
        max_tokens: int = 4096,
        system: str = "",
    ) -> dict:
        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)

        for block in response.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", "") == tool_name
            ):
                return block.input

        raise RuntimeError(f"Model did not call expected tool: {tool_name}")
