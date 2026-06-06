# ---------------------------------------------------------------------------
# llm_wrapper/llm.py
# ---------------------------------------------------------------------------
# Single entry point for all LLM calls.
#
# Provider resolution order (first match wins):
#   1. provider= argument passed explicitly to call_llm / call_llm_with_tool
#   2. inferred from model name prefix (claude-* → anthropic, gpt-* → openai, etc.)
#   3. LLM_PROVIDER environment variable
#   4. "anthropic" as last-resort default
#
# Provider instances are cached — safe to use multiple providers in one process.
#
# Other environment variables:
#   LOG_LLM_CALLS — set "1" to enable JSONL logging (default: off)
#   LLM_LOGS_DIR  — directory for llm_calls.jsonl (default: ~/.llm_wrapper/logs)
# ---------------------------------------------------------------------------

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from llm_wrapper.providers.base_provider import BaseProvider

load_dotenv()

LOG_LLM_CALLS = os.getenv("LOG_LLM_CALLS", "0") == "1"
LOGS_DIR      = Path(os.getenv("LLM_LOGS_DIR", str(Path.home() / ".llm_wrapper" / "logs")))

LOGS_DIR.mkdir(parents=True, exist_ok=True)

_cache: dict[str, BaseProvider] = {}


def _make_provider(name: str) -> BaseProvider:
    if name == "anthropic":
        from llm_wrapper.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    if name == "openai":
        from llm_wrapper.providers.openai_provider import OpenAIProvider
        return OpenAIProvider()
    if name == "ollama":
        from llm_wrapper.providers.ollama_provider import OllamaProvider
        return OllamaProvider()
    if name == "gemini":
        from llm_wrapper.providers.gemini_provider import GeminiProvider
        return GeminiProvider()
    if name == "fake":
        from llm_wrapper.providers.fake_provider import FakeProvider
        return FakeProvider()
    raise ValueError(f"Unknown provider: {name!r}")


def _get_provider(name: str) -> BaseProvider:
    if name not in _cache:
        _cache[name] = _make_provider(name)
    return _cache[name]


def _infer_provider(model: str) -> str | None:
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    if model.startswith("gemini"):
        return "gemini"
    return None


def _resolve_provider(provider: str | None, model: str) -> str:
    return provider or _infer_provider(model) or os.getenv("LLM_PROVIDER") or "anthropic"


def _log_call(call_type: str, provider: str, model: str, system: str,
              prompt: str, response: str, tool_name: str = None) -> None:
    if not LOG_LLM_CALLS:
        return
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "call_type": call_type,
        "provider": provider,
        "model": model,
        "system": system,
        "prompt": prompt,
        "response": response,
    }
    if tool_name:
        entry["tool_name"] = tool_name
    with open(LOGS_DIR / "llm_calls.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def call_llm(
    prompt: str,
    model: str,
    provider: str = None,
    max_tokens: int = 4096,
    system: str = "",
) -> str:
    resolved = _resolve_provider(provider, model)
    response = _get_provider(resolved).call_text(
        prompt=prompt, model=model, max_tokens=max_tokens, system=system,
    )
    _log_call("text", resolved, model, system, prompt, response)
    return response


def call_llm_with_tool(
    prompt: str,
    tool_schema: dict,
    tool_name: str,
    model: str,
    provider: str = None,
    max_tokens: int = 4096,
    system: str = "",
) -> dict:
    resolved = _resolve_provider(provider, model)
    response = _get_provider(resolved).call_tool(
        prompt=prompt, tool_schema=tool_schema, tool_name=tool_name,
        model=model, max_tokens=max_tokens, system=system,
    )
    _log_call("tool", resolved, model, system, prompt, json.dumps(response), tool_name)
    return response
