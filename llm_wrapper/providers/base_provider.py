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
