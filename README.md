# llm-wrapper

Provider-agnostic LLM call layer. Supports Anthropic, OpenAI, Gemini, and Ollama.
Enforces structured output via native tool-calling — no fragile `json.loads()` on free text.

## Install

```bash
pip install git+https://github.com/agentic-writing/llm_wrapper.git
```

Gemini requires one extra package:
```bash
pip install google-genai
```

## Setup

The only thing required in `.env` is your API key:

```bash
ANTHROPIC_API_KEY=your_key_here
```

Copy `.env.example` for all available options:
```bash
cp .env.example .env
```

Verify everything works:
```bash
python -m llm_wrapper
```

## Usage

Provider and model can be set in `.env` or passed explicitly as arguments. Explicit arguments always win.

**Text call:**
```python
from llm_wrapper.llm import call_llm

response = call_llm(
    prompt="Summarise this paragraph: ...",
    model="claude-haiku-4-5",
    provider="anthropic",   # optional — inferred from model name if omitted
)
```

**Tool call** (schema-enforced, always returns a plain dict):
```python
from llm_wrapper.llm import call_llm_with_tool

schema = {
    "name": "classify",
    "description": "Classify the sentiment of the input.",
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        },
        "required": ["label"],
    },
}

result = call_llm_with_tool(
    prompt="I absolutely loved this product!",
    tool_schema=schema,
    tool_name="classify",
    model="claude-haiku-4-5",
    provider="anthropic",   # optional
)
# result → {"label": "positive"}
```

Some Anthropic models reject a forced `tool_choice` with a 400 error. For those, the provider falls back to `tool_choice: auto` and adds an instruction to the system prompt to call the tool. If the model still answers in text, `call_llm_with_tool` raises `RuntimeError`, the same as any missing tool call.

**Images** (both call types, all providers):
```python
with open("plot.png", "rb") as f:
    image = f.read()

result = call_llm_with_tool(
    prompt="Which way does the curve in this image bend?",
    tool_schema=schema,
    tool_name="report_bend",
    model="claude-haiku-4-5",
    images=[image],   # list of PNG bytes
)
```

- Only PNG is accepted. Anything else raises `ValueError` before a request is sent.
- Images are placed ahead of the prompt in the user message. Ollama sends them in a separate `images` field, so it has no text-relative order.
- Ollama raises `RuntimeError` if the model doesn't report `vision` support, or if that can't be confirmed. Calls without images never make this check.
- With `LOG_LLM_CALLS=1`, the log records `image_count` but never the image bytes.

## Provider resolution

When `provider` is not passed explicitly, it is resolved in this order:

1. Inferred from `model` name (`claude-*` → anthropic, `gpt-*` → openai, `gemini-*` → gemini)
2. `LLM_PROVIDER` environment variable
3. `"anthropic"` as default

Provider instances are cached — safe to use multiple providers in one process.

## `.env` reference

See `.env.example` for all options. Commonly used:

| Variable | Notes |
|---|---|
| `ANTHROPIC_API_KEY` | Required for Anthropic |
| `OPENAI_API_KEY` | Required for OpenAI |
| `GEMINI_API_KEY` | Required for Gemini |
| `LLM_PROVIDER` | Recommended when model name is ambiguous (e.g. Ollama local models) |
| `MODEL` | Used by `python -m llm_wrapper` smoke test only |
| `LOG_LLM_CALLS` | Set `1` to log all calls to `LLM_LOGS_DIR/llm_calls.jsonl` |
| `OLLAMA_BASE_URL` | Default: `http://localhost:11434` |

## Adding a provider

See `CONTRIBUTING.md`.
