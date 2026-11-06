"""Upstream (provider-side) failures.

These are INTERNAL exceptions. Their messages may contain provider detail
and are only ever logged; the gateway maps them to `GatewayError`s before
anything reaches a client.
"""

from __future__ import annotations

from enum import Enum


class FailureKind(str, Enum):
    RATE_LIMITED = "rate_limited"  # HTTP 429
    TIMEOUT = "timeout"  # attempt deadline exceeded / 408 / 504
    SERVER_ERROR = "server_error"  # 5xx
    CONNECTION_ERROR = "connection_error"  # DNS, TLS, reset
    BAD_REQUEST = "bad_request"  # other 4xx
    UNKNOWN = "unknown"  # adapter bug / unexpected exception


class UpstreamError(Exception):
    kind: FailureKind = FailureKind.UNKNOWN

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code

    @staticmethod
    def from_status(status_code: int, message: str, *, provider: str) -> "UpstreamError":
        """Classify an HTTP status from a provider into the right subclass."""
        if status_code == 429:
            cls: type[UpstreamError] = UpstreamRateLimitError
        elif status_code in (408, 504):
            cls = UpstreamTimeoutError
        elif 400 <= status_code < 500:
            cls = UpstreamBadRequestError
        else:
            cls = UpstreamServerError
        return cls(message, provider=provider, status_code=status_code)


class UpstreamRateLimitError(UpstreamError):
    kind = FailureKind.RATE_LIMITED


class UpstreamTimeoutError(UpstreamError):
    kind = FailureKind.TIMEOUT


class UpstreamServerError(UpstreamError):
    kind = FailureKind.SERVER_ERROR


class UpstreamConnectionError(UpstreamError):
    kind = FailureKind.CONNECTION_ERROR


class UpstreamBadRequestError(UpstreamError):
    kind = FailureKind.BAD_REQUEST
