"""Typed, validated configuration for the gateway."""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_gateway.providers.errors import FailureKind


@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    """Sliding-window token budget applied per API key."""

    tokens_per_window: int = 50_000
    window_seconds: float = 60.0
    # How often the background sweeper purges expired rows for *all* keys.
    sweep_interval_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.tokens_per_window <= 0:
            raise ValueError("tokens_per_window must be positive")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.sweep_interval_seconds <= 0:
            raise ValueError("sweep_interval_seconds must be positive")


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    """Failover behaviour across the provider chain."""

    # Per-attempt deadline. A provider that has not answered by then is
    # cancelled and the next provider in the chain is tried.
    attempt_timeout_seconds: float = 3.0
    # Upstream failure kinds that trigger failover. Anything else (e.g. a
    # 400 from the provider) fails fast, because retrying elsewhere would
    # not help and would double-spend.
    failover_on: frozenset[FailureKind] = frozenset(
        {FailureKind.RATE_LIMITED, FailureKind.TIMEOUT}
    )

    def __post_init__(self) -> None:
        if self.attempt_timeout_seconds <= 0:
            raise ValueError("attempt_timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class RequestLimits:
    """Input validation bounds for incoming completion requests."""

    max_prompt_chars: int = 200_000
    max_output_tokens: int = 8_192
    default_output_tokens: int = 256


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    database_path: str = "data/gateway.db"
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    request_limits: RequestLimits = field(default_factory=RequestLimits)
