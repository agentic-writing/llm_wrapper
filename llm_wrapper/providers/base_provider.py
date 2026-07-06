# ---------------------------------------------------------------------------
# llm_wrapper/providers/base_provider.py
# ---------------------------------------------------------------------------
# Abstract interface that all LLM providers must implement.
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod


class BaseProvider(ABC):

    @abstractmethod
    def call_text(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        system: str,
    ) -> str: ...

    @abstractmethod
    def call_tool(
        self,
        prompt: str,
        tool_schema: dict,
        tool_name: str,
        model: str,
        max_tokens: int,
        system: str,
    ) -> dict: ...

    @staticmethod
    def _raise_if_truncated(truncated: bool, max_tokens: int) -> None:
        """
        Raise a clear, consistent error when a provider's response was cut off at
        max_tokens, instead of letting a truncated response silently fail to parse
        (or return partial text) downstream with no indication of the real cause.
        """
        if truncated:
            raise RuntimeError(
                f"Response truncated: hit max_tokens={max_tokens}. Increase max_tokens to avoid this."
            )
