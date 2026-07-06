# ---------------------------------------------------------------------------
# llm_wrapper/providers/ollama_provider.py
# ---------------------------------------------------------------------------
# Ollama implementation of BaseProvider.
# Uses Ollama's native /api/chat endpoint.
#
# Important:
# call_tool() checks (and caches) each model's capabilities via /api/show; for models
# that report "tools" support (e.g. qwen2.5) it uses Ollama's native tool-calling
# (tools=[...] on /api/chat). Models without tool support (e.g. phi4) skip that
# attempt entirely, since Ollama rejects tools=[...] outright (HTTP 400) for them.
# Either way, if the native attempt doesn't produce a schema-conformant tool call,
# call_tool() falls back to emulating tool use by asking the model to return strict
# JSON matching the requested tool schema.
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
        self._tool_support_cache: dict[str, bool] = {}

    def _supports_tools(self, model: str) -> bool:
        """
        Check (and cache) whether `model` advertises native tool-calling support,
        via POST /api/show. Checked once per model per process — avoids sending a
        tools=[...] request that the server is guaranteed to reject for models
        that don't support it (e.g. phi4). If the capability check itself fails
        (network hiccup, older Ollama without /api/show details), assume no
        support — the JSON-prompt fallback is always safe, just slower.
        """
        if model not in self._tool_support_cache:
            try:
                response = requests.post(
                    f"{self.base_url}/api/show",
                    json={"model": model},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                capabilities = response.json().get("capabilities", [])
                self._tool_support_cache[model] = "tools" in capabilities
            except Exception:
                self._tool_support_cache[model] = False
        return self._tool_support_cache[model]

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
        self._raise_if_truncated(data.get("done_reason") == "length", max_tokens)
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
        """
        Use Ollama's native tool-calling (POST /api/chat with tools=[...]) for models
        that support it (capabilities include "tools", checked once via _supports_tools)
        — the server parses the model's structured tool call itself, which is far more
        reliable than asking the model to freehand a bare JSON blob. Models without tool
        support (e.g. phi4) skip the native attempt entirely — Ollama rejects a tools=[...]
        request outright (HTTP 400) for those, so trying would only waste a round trip.
        Ollama doesn't validate arguments against the schema server-side, so a returned
        tool call is only accepted if it has all of the schema's required fields.
        Any failure in the native path (missing/wrong fields, no tool call, request
        error) falls back to prompt-based JSON emulation.
        """
        if self._supports_tools(model):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": model,
                "messages": messages,
                "tools": [self._to_ollama_tool(tool_schema, tool_name)],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": max_tokens,
                },
            }

            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                native_data = response.json()
                self._raise_if_truncated(native_data.get("done_reason") == "length", max_tokens)
                tool_calls = native_data.get("message", {}).get("tool_calls")

                if tool_calls:
                    arguments = tool_calls[0]["function"]["arguments"]
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                    required = tool_schema.get("input_schema", {}).get("required", [])
                    if all(key in arguments for key in required):
                        return arguments
                    # Model called the tool but with the wrong/missing fields (schema
                    # isn't enforced server-side) — fall through to JSON-prompt path.
            except Exception:
                # Native tool-calling errored (server rejection, network issue, etc.)
                # — fall through to the JSON-prompt path rather than crashing.
                pass

        return self._call_tool_via_json_prompt(
            prompt=prompt,
            tool_schema=tool_schema,
            tool_name=tool_name,
            model=model,
            max_tokens=max_tokens,
            system=system,
        )

    def _call_tool_via_json_prompt(
        self,
        prompt: str,
        tool_schema: dict,
        tool_name: str,
        model: str,
        max_tokens: int,
        system: str,
    ) -> dict:
        """Fallback for models/servers without native tool-calling: prompt for raw JSON."""
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
    def _to_ollama_tool(tool_schema: dict, tool_name: str) -> dict:
        """Convert an Anthropic-style tool schema to Ollama/OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": tool_schema.get("name", tool_name),
                "description": tool_schema.get("description", ""),
                "parameters": tool_schema.get("input_schema", {}),
            },
        }

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
