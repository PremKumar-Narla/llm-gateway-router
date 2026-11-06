"""Provider adapter contract.

A real adapter (OpenAI, Anthropic, Bedrock, ...) implements `complete()` and
translates transport failures into `UpstreamError` subclasses, typically via
`UpstreamError.from_status(...)`. Adapters must be cancellation-safe: the
router cancels them when the attempt deadline passes, so they must not
swallow `asyncio.CancelledError`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_gateway.models import CompletionRequest, ProviderCompletion


class ModelProvider(ABC):
    def __init__(self, name: str, model: str) -> None:
        self.name = name
        self.model = model

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> ProviderCompletion:
        """Return a completion or raise an `UpstreamError`."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, model={self.model!r})"
