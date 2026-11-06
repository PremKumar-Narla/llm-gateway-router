"""Domain objects passed between gateway layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from llm_gateway.config import RequestLimits
from llm_gateway.errors import InvalidRequestError


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated caller. `key_id` is a hash; the raw key is never kept."""

    key_id: str
    tenant_id: str
    # Optional per-key override of the default token budget.
    tokens_per_window: int | None = None


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    prompt: str
    max_tokens: int
    temperature: float = 0.7

    @classmethod
    def from_payload(cls, payload: Any, limits: RequestLimits) -> "CompletionRequest":
        """Validate untrusted JSON input. Error messages are client-safe."""
        if not isinstance(payload, Mapping):
            raise InvalidRequestError("Request body must be a JSON object.")

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvalidRequestError("'prompt' must be a non-empty string.")
        if len(prompt) > limits.max_prompt_chars:
            raise InvalidRequestError(
                f"'prompt' exceeds {limits.max_prompt_chars} characters."
            )

        max_tokens = payload.get("max_tokens", limits.default_output_tokens)
        # bool is a subclass of int; reject it explicitly.
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise InvalidRequestError("'max_tokens' must be an integer.")
        if not 1 <= max_tokens <= limits.max_output_tokens:
            raise InvalidRequestError(
                f"'max_tokens' must be between 1 and {limits.max_output_tokens}."
            )

        temperature = payload.get("temperature", 0.7)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise InvalidRequestError("'temperature' must be a number.")
        if not 0.0 <= float(temperature) <= 2.0:
            raise InvalidRequestError("'temperature' must be between 0 and 2.")

        return cls(prompt=prompt, max_tokens=max_tokens, temperature=float(temperature))


@dataclass(frozen=True, slots=True)
class ProviderCompletion:
    """What a provider adapter returns on success."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class RoutingAttempt:
    """One provider call. Internal: logged and audited, never sent to clients."""

    provider: str
    outcome: str  # "success" or a FailureKind value
    latency_ms: float


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    request_id: str
    completion: ProviderCompletion
    fallback_used: bool
    attempts: tuple[RoutingAttempt, ...] = field(default=())

    def to_public_dict(self) -> dict[str, Any]:
        """Client-facing shape. Provider names and attempt details stay internal."""
        return {
            "id": self.request_id,
            "object": "text_completion",
            "model": self.completion.model,
            "text": self.completion.text,
            "fallback_used": self.fallback_used,
            "usage": {
                "prompt_tokens": self.completion.prompt_tokens,
                "completion_tokens": self.completion.completion_tokens,
                "total_tokens": self.completion.total_tokens,
            },
        }
