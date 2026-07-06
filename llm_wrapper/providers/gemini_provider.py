# ---------------------------------------------------------------------------
# llm_wrapper/providers/gemini_provider.py
# ---------------------------------------------------------------------------
# Google Gemini implementation of BaseProvider.
# Uses the google-genai SDK (Google AI Studio / Gemini Developer API).
#
# Requires: pip install "llm-wrapper[gemini]"
# API key:  GEMINI_API_KEY  (or GOOGLE_API_KEY as fallback)
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Any

from llm_wrapper.providers.base_provider import BaseProvider


class GeminiProvider(BaseProvider):

    def __init__(self):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError(
                "google-genai is not installed. Run: pip install google-genai"
            )
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) missing")
        self.client = genai.Client(api_key=api_key)
        self._types = types

    def call_text(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        system: str = "",
    ) -> str:
        config = self._types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=max_tokens,
        )
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        finish_reason = getattr(response.candidates[0], "finish_reason", None) if response.candidates else None
        self._raise_if_truncated(finish_reason == "MAX_TOKENS", max_tokens)
        return (response.text or "").strip()

    def call_tool(
        self,
        prompt: str,
        tool_schema: dict,
        tool_name: str,
        model: str,
        max_tokens: int = 4096,
        system: str = "",
    ) -> dict[str, Any]:
        types = self._types
        declaration = types.FunctionDeclaration(
            name=tool_name,
            description=tool_schema.get("description", ""),
            parameters=tool_schema.get("input_schema", {}),
        )
        tool = types.Tool(function_declarations=[declaration])
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            tools=[tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
            max_output_tokens=max_tokens,
        )
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        finish_reason = getattr(response.candidates[0], "finish_reason", None) if response.candidates else None
        self._raise_if_truncated(finish_reason == "MAX_TOKENS", max_tokens)

        for candidate in response.candidates:
            for part in candidate.content.parts:
                fc = getattr(part, "function_call", None)
                if fc and fc.name == tool_name:
                    return dict(fc.args)

        raise RuntimeError(f"Model did not call expected tool: {tool_name}")
