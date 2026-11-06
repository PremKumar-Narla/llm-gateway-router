"""Ordered failover across model providers with a hard per-attempt deadline.

Timeout handling
----------------
Each attempt runs inside `asyncio.timeout()` *in the caller's task*. When
the deadline passes, the provider coroutine is cancelled and the cancellation
is awaited before we move on, so:

* the primary is fully torn down before the secondary starts -- no orphaned
  tasks, and never two providers generating (and billing) at once;
* a result that lands in the same loop tick as the deadline is discarded and
  treated as a timeout, so the outcome is decided exactly once;
* only *our* deadline is converted to a timeout. A cancellation coming from
  outside (client disconnect, server shutdown) propagates as
  `CancelledError` and is never mistaken for a failover trigger.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from llm_gateway.config import RoutingConfig
from llm_gateway.models import CompletionRequest, ProviderCompletion, RoutingAttempt
from llm_gateway.providers.base import ModelProvider
from llm_gateway.providers.errors import FailureKind, UpstreamError, UpstreamTimeoutError

logger = logging.getLogger("llm_gateway.routing")


@dataclass(frozen=True, slots=True)
class RoutedCompletion:
    completion: ProviderCompletion
    provider: str
    attempts: tuple[RoutingAttempt, ...]

    @property
    def fallback_used(self) -> bool:
        return len(self.attempts) > 1


class RoutingFailedError(Exception):
    """Every eligible provider failed. Internal; mapped by the gateway."""

    def __init__(self, final_kind: FailureKind, attempts: tuple[RoutingAttempt, ...]) -> None:
        super().__init__(f"routing failed: {final_kind.value} after {len(attempts)} attempt(s)")
        self.final_kind = final_kind
        self.attempts = attempts


class FallbackRouter:
    def __init__(self, providers: Sequence[ModelProvider], config: RoutingConfig) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self._providers = tuple(providers)
        self._config = config

    @property
    def providers(self) -> tuple[ModelProvider, ...]:
        return self._providers

    async def route(self, request: CompletionRequest) -> RoutedCompletion:
        attempts: list[RoutingAttempt] = []
        last = len(self._providers) - 1

        for index, provider in enumerate(self._providers):
            started = time.perf_counter()
            try:
                completion = await self._attempt(provider, request)
            except UpstreamError as exc:
                attempts.append(RoutingAttempt(provider.name, exc.kind.value, _ms_since(started)))
                # Full upstream detail is logged here and goes nowhere else.
                logger.warning(
                    "provider attempt failed provider=%s kind=%s status=%s detail=%s",
                    provider.name, exc.kind.value, exc.status_code, exc,
                )
                eligible = exc.kind in self._config.failover_on
                if index == last or not eligible:
                    raise RoutingFailedError(exc.kind, tuple(attempts)) from exc
                logger.info("failing over from %s to %s", provider.name,
                            self._providers[index + 1].name)
                continue

            attempts.append(RoutingAttempt(provider.name, "success", _ms_since(started)))
            return RoutedCompletion(completion, provider.name, tuple(attempts))

        raise AssertionError("unreachable")  # loop always returns or raises

    async def _attempt(self, provider: ModelProvider, request: CompletionRequest) -> ProviderCompletion:
        timeout = self._config.attempt_timeout_seconds
        try:
            async with asyncio.timeout(timeout):
                return await provider.complete(request)
        except TimeoutError as exc:
            # Raised only when *our* deadline fired (or the adapter's own
            # socket timeout). External cancellation is a CancelledError and
            # is deliberately not caught.
            raise UpstreamTimeoutError(
                f"{provider.name} did not respond within {timeout}s", provider=provider.name
            ) from exc
        except UpstreamError:
            raise
        except Exception as exc:
            # Adapter bug or unexpected payload: classify it so the router
            # still produces a controlled outcome.
            raise UpstreamError(
                f"{provider.name} adapter raised {type(exc).__name__}: {exc}",
                provider=provider.name,
            ) from exc


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
