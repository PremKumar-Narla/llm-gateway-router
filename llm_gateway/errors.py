"""Client-facing gateway errors and the sanitizer that renders them.

Rule: every message in this module is authored by the gateway. Upstream
exception text, stack traces, provider names and hostnames are logged
server-side against the request id, and never copied into a GatewayError.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("llm_gateway.errors")


class ErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    UNAUTHENTICATED = "unauthenticated"
    NOT_FOUND = "not_found"
    REQUEST_TOO_LARGE = "request_too_large"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UPSTREAM_REJECTED = "upstream_rejected_request"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    UPSTREAM_TIMEOUT = "upstream_timeout"
    UPSTREAM_ERROR = "upstream_error"
    INTERNAL_ERROR = "internal_error"


class GatewayError(Exception):
    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    http_status: int = 500
    default_message: str = "An internal error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        self.public_message = message or self.default_message
        self.retry_after_seconds = retry_after_seconds
        super().__init__(self.public_message)


class InvalidRequestError(GatewayError):
    code, http_status = ErrorCode.INVALID_REQUEST, 400
    default_message = "The request is invalid."


class UnauthenticatedError(GatewayError):
    code, http_status = ErrorCode.UNAUTHENTICATED, 401
    default_message = "A valid API key is required."


class NotFoundError(GatewayError):
    code, http_status = ErrorCode.NOT_FOUND, 404
    default_message = "Route not found."


class RequestTooLargeError(GatewayError):
    code, http_status = ErrorCode.REQUEST_TOO_LARGE, 413
    default_message = "The request exceeds the per-window token budget."


class RateLimitExceededError(GatewayError):
    code, http_status = ErrorCode.RATE_LIMIT_EXCEEDED, 429
    default_message = "Token rate limit exceeded for this API key."


class ProviderRejectedError(GatewayError):
    code, http_status = ErrorCode.UPSTREAM_REJECTED, 400
    default_message = "The model provider rejected the request."


class ProvidersUnavailableError(GatewayError):
    code, http_status = ErrorCode.UPSTREAM_UNAVAILABLE, 503
    default_message = "All model providers are currently unavailable. Please retry."


class GatewayTimeoutError(GatewayError):
    code, http_status = ErrorCode.UPSTREAM_TIMEOUT, 504
    default_message = "The model providers did not respond in time. Please retry."


class BadGatewayError(GatewayError):
    code, http_status = ErrorCode.UPSTREAM_ERROR, 502
    default_message = "The model provider returned an error."


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    status: int
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)


class ErrorSanitizer:
    """Converts *any* exception into the standard gateway error payload.

        {"error": {"code", "message", "request_id", ["retry_after_seconds"]}}
    """

    def to_response(self, exc: BaseException, request_id: str) -> ErrorResponse:
        if not isinstance(exc, GatewayError):
            # Unknown failure: full detail goes to logs only.
            logger.error(
                "unhandled error request_id=%s", request_id, exc_info=exc
            )
            exc = GatewayError()
        elif exc.__cause__ is not None:
            # A mapped upstream/internal error. Keep the root cause in logs.
            logger.warning(
                "request_id=%s code=%s cause=%r",
                request_id,
                exc.code.value,
                exc.__cause__,
            )

        error: dict[str, Any] = {
            "code": exc.code.value,
            "message": exc.public_message,
            "request_id": request_id,
        }
        headers: dict[str, str] = {}
        if exc.retry_after_seconds is not None:
            retry_after = round(max(0.0, exc.retry_after_seconds), 3)
            error["retry_after_seconds"] = retry_after
            headers["Retry-After"] = str(max(1, math.ceil(retry_after)))

        return ErrorResponse(status=exc.http_status, body={"error": error}, headers=headers)
