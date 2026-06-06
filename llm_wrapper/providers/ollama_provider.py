# ---------------------------------------------------------------------------
# llm_wrapper/providers/ollama_provider.py
# ---------------------------------------------------------------------------
# Ollama implementation of BaseProvider.
# Uses Ollama's native /api/chat endpoint.
#
# Important:
# Ollama does not behave like Anthropic/OpenAI tool calling in a uniform way
# across all local models, so call_tool() emulates tool use by asking the model
# to return strict JSON matching the requested tool schema.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from llm_wrapper.providers.base_provider import BaseProvider


class OllamaProvider(BaseProvider):

    def __init__(self):
        self.base_url    = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434").rstrip("/")
        self.timeout     = int(os.getenv("OLLAMA_TIMEOUT", "60"))
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

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

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        return (data.get("message", {}).get("content") or "").strip()

    def call_tool(
        self,
        prompt: str,
        tool_schema: dict,
        tool_name: str,
        model: str,
        max_tokens: int = 4096,
        system: str = "",
    ) -> dict:
        json_prompt = self._build_json_tool_prompt(
            prompt=prompt,
            tool_schema=tool_schema,
            tool_name=tool_name,
        )

        raw = self.call_text(
            prompt=json_prompt,
            model=model,
            max_tokens=max_tokens,
            system=system,
        )

        try:
            return self._parse_json(raw)
        except Exception:
            # One repair attempt.
            repair_prompt = f"""
                The previous response was not valid JSON.

                Return ONLY valid JSON for tool `{tool_name}`.
                Do not include markdown, explanation, or code fences.

                Tool schema:
                {json.dumps(tool_schema.get("input_schema", {}), indent=2)}

                Previous invalid response:
                {raw}
                """.strip()

            raw2 = self.call_text(
                prompt=repair_prompt,
                model=model,
                max_tokens=max_tokens,
                system=system,
            )

            return self._parse_json(raw2)

    @staticmethod
    def _build_json_tool_prompt(
        prompt: str,
        tool_schema: dict,
        tool_name: str,
    ) -> str:
        input_schema = tool_schema.get("input_schema", {})

        return f"""
            You must respond as if calling the tool `{tool_name}`.

            Return ONLY a valid JSON object matching this schema.
            Do not include markdown.
            Do not include ```json fences.
            Do not include explanations.

            JSON schema:
            {json.dumps(input_schema, indent=2)}

            User prompt:
            {prompt}
            """.strip()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()

        # Remove common markdown fences if the model ignores instructions.
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: extract first JSON object from the response.
            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise

            return json.loads(cleaned[start:end + 1])
