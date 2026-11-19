import json
import os
from llm_wrapper.llm import call_llm, call_llm_with_tool, _resolve_provider

MODEL    = os.getenv("MODEL", "claude-haiku-4-5")
PROVIDER = _resolve_provider(os.getenv("LLM_PROVIDER"), MODEL)
EFFORT   = os.getenv("EFFORT") or None  # e.g. "low" or "high" (Anthropic only); unset = model default

# Room for models that think before answering; thinking tokens count toward max_tokens.
MAX_TOKENS = 2000

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

print(f"Provider: {PROVIDER}  |  Model: {MODEL}  |  Effort: {EFFORT or 'model default'}\n")

text = call_llm("Reply with exactly: hello", model=MODEL, provider=PROVIDER, max_tokens=MAX_TOKENS,
                effort=EFFORT)
print(f"call_llm:           {text!r}")

result = call_llm_with_tool(
    prompt="I absolutely loved this product!",
    tool_schema=_SCHEMA,
    tool_name="classify_sentiment",
    model=MODEL,
    provider=PROVIDER,
    max_tokens=MAX_TOKENS,
    effort=EFFORT,
)
print(f"call_llm_with_tool: {json.dumps(result)}")
