# ---------------------------------------------------------------------------
# llm_wrapper/providers/base_provider.py
# ---------------------------------------------------------------------------
# Abstract interface that all LLM providers must implement.
# ---------------------------------------------------------------------------

import base64
from abc import ABC, abstractmethod

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class BaseProvider(ABC):

    @abstractmethod
    def call_text(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        system: str,
        images: list[bytes] | None = None,
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
        images: list[bytes] | None = None,
    ) -> dict: ...

    @staticmethod
    def _check_images(images: list[bytes] | None) -> list[bytes]:
        """
        Return images as a list (empty for None), raising ValueError unless every
        item is PNG bytes. Providers label images as image/png, so anything else is
        rejected here rather than sent with the wrong media type.
        """
        checked = list(images or [])
        for i, image in enumerate(checked):
            if not isinstance(image, (bytes, bytearray)) or bytes(image[:8]) != _PNG_SIGNATURE:
                raise ValueError(f"images[{i}] is not PNG bytes.")
        return checked

    @classmethod
    def _images_to_base64(cls, images: list[bytes] | None) -> list[str]:
        """Validate images and return each one as a base64 string."""
        return [base64.b64encode(image).decode("ascii") for image in cls._check_images(images)]

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
