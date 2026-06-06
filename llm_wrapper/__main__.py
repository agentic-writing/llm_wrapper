import json
import os
from llm_wrapper.llm import call_llm, call_llm_with_tool, _resolve_provider

MODEL    = os.getenv("MODEL", "claude-haiku-4-5")
PROVIDER = _resolve_provider(os.getenv("LLM_PROVIDER"), MODEL)

_SCHEMA = {
    "name": "classify_sentiment",
    "description": "Classify the sentiment of the input text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        },
        "required": ["label"],
    },
}

print(f"Provider: {PROVIDER}  |  Model: {MODEL}\n")

text = call_llm("Reply with exactly: hello", model=MODEL, provider=PROVIDER, max_tokens=16)
print(f"call_llm:           {text!r}")

result = call_llm_with_tool(
    prompt="I absolutely loved this product!",
    tool_schema=_SCHEMA,
    tool_name="classify_sentiment",
    model=MODEL,
    provider=PROVIDER,
    max_tokens=64,
)
print(f"call_llm_with_tool: {json.dumps(result)}")
