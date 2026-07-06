# ---------------------------------------------------------------------------
# llm_wrapper/providers/openai_provider.py
# ---------------------------------------------------------------------------
# OpenAI implementation of BaseProvider. Wraps the OpenAI Python SDK.
# Internally converts Anthropic-style tool schemas (input_schema key) to
# OpenAI function-calling format (parameters key) so existing agents work
# without modification when LLM_PROVIDER=openai is set.
# ---------------------------------------------------------------------------

import json
import os

from openai import OpenAI

from llm_wrapper.providers.base_provider import BaseProvider


class OpenAIProvider(BaseProvider):

    def __init__(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY missing")
        self.client = OpenAI()

    @staticmethod
    def _to_openai_tool(tool_schema: dict) -> dict:
        """
        Convert an Anthropic-style tool schema to OpenAI function format.

        Anthropic: {"name": ..., "description": ..., "input_schema": {...}}
        OpenAI:    {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
        """
        return {
            "type": "function",
            "function": {
                "name": tool_schema["name"],
                "description": tool_schema.get("description", ""),
                "parameters": tool_schema.get("input_schema", {}),
            },
        }

    def call_text(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        system: str = "",
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=messages,
        )
        self._raise_if_truncated(response.choices[0].finish_reason == "length", max_tokens)

        return (response.choices[0].message.content or "").strip()

    def call_tool(
        self,
        prompt: str,
        tool_schema: dict,
        tool_name: str,
        model: str,
        max_tokens: int = 4096,
        system: str = "",
    ) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            tools=[self._to_openai_tool(tool_schema)],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            messages=messages,
        )

        self._raise_if_truncated(response.choices[0].finish_reason == "length", max_tokens)

        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise RuntimeError(f"Model did not call expected tool: {tool_name}")

        return json.loads(tool_calls[0].function.arguments)
