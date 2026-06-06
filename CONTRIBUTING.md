# Adding a New Provider

Four steps. No other files change.

---

## 1. Create the provider file

Create `llm_wrapper/providers/my_provider.py` and inherit from `BaseProvider`:

```python
import os
from llm_wrapper.providers.base_provider import BaseProvider


class MyProvider(BaseProvider):

    def __init__(self):
        api_key = os.getenv("MY_PROVIDER_API_KEY")
        if not api_key:
            raise RuntimeError("MY_PROVIDER_API_KEY missing")
        # initialise the SDK client here

    def call_text(self, prompt: str, model: str, max_tokens: int = 4096, system: str = "") -> str:
        # make the API call, return a plain string
        ...

    def call_tool(self, prompt: str, tool_schema: dict, tool_name: str,
                  model: str, max_tokens: int = 4096, system: str = "") -> dict:
        # force the model to call the tool, return a plain dict
        # tool_schema uses Anthropic format: {"name", "description", "input_schema"}
        ...
```

**`call_tool` contract:** must return a `dict` matching `tool_schema["input_schema"]`. Never return a string or raise on a well-formed schema. If the SDK doesn't support native tool calling, emulate it by asking the model to return strict JSON (see `ollama_provider.py` for the pattern).

**Optional SDK dependency:** if the SDK is not a core requirement, import it inside `__init__` with a helpful error:

```python
def __init__(self):
    try:
        import my_sdk
    except ImportError:
        raise RuntimeError("my-sdk not installed. Run: pip install my-sdk")
    ...
```

---

## 2. Register the provider name

In `llm_wrapper/llm.py`, add one branch to `_make_provider`:

```python
if name == "my-provider":
    from llm_wrapper.providers.my_provider import MyProvider
    return MyProvider()
```

---

## 3. Update auto-detection (if applicable)

If the provider's model names follow a recognisable prefix, add a rule to `_infer_provider` in `llm_wrapper/llm.py`:

```python
if model.startswith("my-prefix-"):
    return "my-provider"
```

---

## 4. Add to `.env.example` and the integration test

In `.env.example`:
```
# MY_PROVIDER_API_KEY=your_key_here
```

In `tests/test_llm.py`, copy one of the existing integration test classes and adjust the model name and skip condition:

```python
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("MY_PROVIDER_API_KEY"), reason="MY_PROVIDER_API_KEY not set")
class TestMyProvider:
    MODEL = "my-default-model"

    def setup_method(self):
        from llm_wrapper.providers.my_provider import MyProvider
        self.provider = MyProvider()

    def test_call_text(self):
        result = self.provider.call_text("Reply with just the word hello.", model=self.MODEL, max_tokens=20)
        assert isinstance(result, str) and len(result) > 0

    def test_call_tool(self):
        result = self.provider.call_tool("I love this product!", _SENTIMENT_SCHEMA, "classify_sentiment", model=self.MODEL, max_tokens=64)
        assert result.get("label") in ("positive", "negative", "neutral")
```

---

## Checklist

- [ ] `providers/my_provider.py` — inherits `BaseProvider`, implements `call_text` + `call_tool`
- [ ] `llm.py` — registered in `_make_provider`
- [ ] `llm.py` — prefix added to `_infer_provider` (if applicable)
- [ ] `.env.example` — API key variable documented
- [ ] `tests/test_llm.py` — integration test class added
