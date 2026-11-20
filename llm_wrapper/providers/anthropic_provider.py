# ---------------------------------------------------------------------------
# llm_wrapper/providers/anthropic_provider.py
# ---------------------------------------------------------------------------
# Anthropic implementation of BaseProvider. Wraps the Anthropic Python SDK
# and hides all API-specific details (response parsing, tool extraction,
# system prompt format) from the rest of the framework.
# ---------------------------------------------------------------------------

import os

from anthropic import Anthropic, BadRequestError

from llm_wrapper.providers.base_provider import BaseProvider


class AnthropicProvider(BaseProvider):

    # Requests with max_tokens above this are streamed (the result is the same Message). The SDK
    # refuses long non-streaming requests (above about 21,000 max_tokens), and a long call
    # without streaming can hit the 10-minute request timeout.
    STREAM_ABOVE_MAX_TOKENS = 16_000

    def __init__(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY missing")
        self.client = Anthropic()
        self._no_forced_tool: set[str] = set()

    def _send(self, kwargs: dict) -> object:
        """messages.create, or a streamed request read to the end for a large max_tokens."""
        if kwargs.get("max_tokens", 0) > self.STREAM_ABOVE_MAX_TOKENS:
            with self.client.messages.stream(**kwargs) as stream:
                return stream.get_final_message()
        return self.client.messages.create(**kwargs)

    def _create_calling_tool(self, kwargs: dict, tool_name: str) -> object:
        """
        Send a request that must call tool_name. Some models reject a forced tool_choice
        with a 400, so those get tool_choice auto plus an explicit instruction instead.
        Rejection is learned once per model per process rather than kept in a model list.
        If the model then answers in text, call_tool raises as for any missing tool call.
        """
        model = kwargs["model"]
        if model not in self._no_forced_tool:
            try:
                return self._send({**kwargs, "tool_choice": {"type": "tool", "name": tool_name}})
            except BadRequestError as error:
                if "tool_choice" not in str(error) or "not supported" not in str(error):
                    raise
                self._no_forced_tool.add(model)

        instruction = f"You must respond by calling the {tool_name} tool."
        system = kwargs.get("system")
        auto_kwargs = {**kwargs, "system": f"{system}\n\n{instruction}" if system else instruction}
        return self._send({**auto_kwargs, "tool_choice": {"type": "auto"}})

    @classmethod
    def _user_content(cls, prompt: str, images: list[bytes] | None) -> str | list[dict]:
        """Plain string without images; otherwise image blocks first, then the text."""
        encoded = cls._images_to_base64(images)
        if not encoded:
            return prompt
        blocks: list[dict] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}
            for data in encoded
        ]
        blocks.append({"type": "text", "text": prompt})
        return blocks

    @staticmethod
    def _usage(response) -> dict | None:
        """
        Token counts of a response: input_tokens, output_tokens (which include thinking
        tokens), cache_creation_input_tokens, cache_read_input_tokens. None if the response
        carries no usage.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return {name: getattr(usage, name, None) for name in (
            "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}

    @staticmethod
    def _add_effort(kwargs: dict, effort: str | None) -> None:
        """
        Send effort as output_config.effort. Without it the request is unchanged and the
        model uses its default effort. The API checks the level, since the levels a model
        accepts differ by model.
        """
        if effort is not None:
            kwargs["output_config"] = {"effort": effort}

    def call_text(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        system: str = "",
        images: list[bytes] | None = None,
        effort: str | None = None,
    ) -> str:
        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": self._user_content(prompt, images)}],
        )
        if system:
            kwargs["system"] = system
        self._add_effort(kwargs, effort)

        response = self._send(kwargs)
        self.last_usage = self._usage(response)  # recorded even if truncated below
        self._raise_if_truncated(response.stop_reason == "max_tokens", max_tokens)

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
        images: list[bytes] | None = None,
        effort: str | None = None,
    ) -> dict:
        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            tools=[tool_schema],
            messages=[{"role": "user", "content": self._user_content(prompt, images)}],
        )
        if system:
            kwargs["system"] = system
        self._add_effort(kwargs, effort)

        response = self._create_calling_tool(kwargs, tool_name)
        self.last_usage = self._usage(response)  # recorded even if truncated below
        self._raise_if_truncated(response.stop_reason == "max_tokens", max_tokens)

        for block in response.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", "") == tool_name
            ):
                return block.input

        raise RuntimeError(f"Model did not call expected tool: {tool_name}")
