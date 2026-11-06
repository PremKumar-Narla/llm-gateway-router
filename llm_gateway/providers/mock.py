"""Deterministic in-process provider for tests, demos and local dev."""

from __future__ import annotations

import asyncio
import random
from collections import deque
from collections.abc import Iterable
from enum import Enum

from llm_gateway.models import CompletionRequest, ProviderCompletion
from llm_gateway.providers.base import ModelProvider
from llm_gateway.providers.errors import UpstreamError
from llm_gateway.tokenizer import HeuristicTokenEstimator


class Behavior(str, Enum):
    SUCCESS = "success"
    RATE_LIMIT = "rate_limit"  # responds with HTTP 429
    HANG = "hang"  # never responds (exercises the attempt timeout)
    SERVER_ERROR = "server_error"  # responds with HTTP 500
    BAD_REQUEST = "bad_request"  # responds with HTTP 400
    CRASH = "crash"  # adapter bug: raises a non-Upstream exception


class MockProvider(ModelProvider):
    """Replays a script of behaviours, then falls back to `default`.

    Upstream error messages deliberately contain "internal" detail so tests
    can assert that none of it leaks into client responses.
    """

    def __init__(
        self,
        name: str,
        model: str,
        *,
        latency_seconds: float = 0.0,
        script: Iterable[Behavior] = (),
        default: Behavior = Behavior.SUCCESS,
        failure_rate: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name, model)
        self.latency_seconds = latency_seconds
        self.default = default
        self.failure_rate = failure_rate
        self._script: deque[Behavior] = deque(script)
        self._rng = random.Random(seed)
        self._estimator = HeuristicTokenEstimator()
        self.calls = 0
        self.cancelled = 0
        self.completed = 0

    def _next_behavior(self) -> Behavior:
        if self._script:
            return self._script.popleft()
        if self.failure_rate and self._rng.random() < self.failure_rate:
            return self._rng.choice([Behavior.RATE_LIMIT, Behavior.HANG])
        return self.default

    async def complete(self, request: CompletionRequest) -> ProviderCompletion:
        self.calls += 1
        behavior = self._next_behavior()
        try:
            if behavior is Behavior.HANG:
                await asyncio.Event().wait()  # until cancelled
            if self.latency_seconds:
                await asyncio.sleep(self.latency_seconds)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise

        detail = f"upstream {self.name} at 10.0.0.12:8443 Traceback (most recent call last): ..."
        if behavior is Behavior.RATE_LIMIT:
            raise UpstreamError.from_status(429, detail, provider=self.name)
        if behavior is Behavior.SERVER_ERROR:
            raise UpstreamError.from_status(500, detail, provider=self.name)
        if behavior is Behavior.BAD_REQUEST:
            raise UpstreamError.from_status(400, detail, provider=self.name)
        if behavior is Behavior.CRASH:
            raise KeyError(f"choices missing in {self.name} payload: {detail}")

        self.completed += 1
        completion_tokens = min(request.max_tokens, 24)
        return ProviderCompletion(
            text=f"[{self.model}] response to: {request.prompt[:40]}",
            model=self.model,
            prompt_tokens=self._estimator.count(request.prompt),
            completion_tokens=completion_tokens,
        )
