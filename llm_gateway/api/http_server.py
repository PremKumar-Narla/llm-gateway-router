"""Minimal stdlib asyncio HTTP/1.1 transport for the gateway.

Routes:
    POST /v1/completions   (Authorization: Bearer <api key>)
    GET  /healthz

One request per connection (`Connection: close`). Header/body sizes and
read time are bounded. In production you would front this with a real
ASGI server; the gateway core does not depend on this module.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from http import HTTPStatus

from llm_gateway.errors import ErrorSanitizer, InvalidRequestError, NotFoundError
from llm_gateway.gateway import GatewayResponse, LLMGateway

logger = logging.getLogger("llm_gateway.http")


class _BadHttpRequest(Exception):
    pass


class GatewayHttpServer:
    def __init__(
        self,
        gateway: LLMGateway,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        max_body_bytes: int = 1_000_000,
        max_header_bytes: int = 16_384,
        read_timeout_seconds: float = 10.0,
    ) -> None:
        self._gateway = gateway
        self._host = host
        self._port = port
        self._max_body = max_body_bytes
        self._max_headers = max_header_bytes
        self._read_timeout = read_timeout_seconds
        self._sanitizer = ErrorSanitizer()
        self._server: asyncio.base_events.Server | None = None

    @property
    def port(self) -> int:
        if self._server is None:
            return self._port
        return self._server.sockets[0].getsockname()[1]

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port, limit=self._max_headers
        )
        logger.info("gateway listening on http://%s:%s", self._host, self.port)

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    # -- connection handling -------------------------------------------------

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            try:
                method, path, headers, body = await asyncio.wait_for(
                    self._read_request(reader), timeout=self._read_timeout
                )
                response = await self._dispatch(method, path, headers, body)
            except (_BadHttpRequest, asyncio.LimitOverrunError, ValueError):
                response = self._error(InvalidRequestError("Malformed HTTP request."))
            except (TimeoutError, asyncio.IncompleteReadError):
                return  # client too slow or went away; nothing to answer
            await self._write_response(writer, response)
        except Exception:
            logger.exception("connection handler failed")
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _read_request(
        self, reader: asyncio.StreamReader
    ) -> tuple[str, str, dict[str, str], bytes]:
        request_line = (await reader.readline()).decode("latin-1").rstrip("\r\n")
        parts = request_line.split(" ")
        if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
            raise _BadHttpRequest()
        method, path = parts[0].upper(), parts[1].split("?", 1)[0]

        headers: dict[str, str] = {}
        total = 0
        while True:
            line = await reader.readline()
            total += len(line)
            if total > self._max_headers:
                raise _BadHttpRequest()
            if line in (b"\r\n", b"\n", b""):
                break
            name, sep, value = line.decode("latin-1").partition(":")
            if not sep:
                raise _BadHttpRequest()
            headers[name.strip().lower()] = value.strip()

        length = int(headers.get("content-length", "0") or 0)
        if length < 0 or length > self._max_body:
            raise _BadHttpRequest()
        body = await reader.readexactly(length) if length else b""
        return method, path, headers, body

    async def _dispatch(
        self, method: str, path: str, headers: dict[str, str], body: bytes
    ) -> GatewayResponse:
        if method == "GET" and path == "/healthz":
            return GatewayResponse(200, {"status": "ok"})
        if method == "POST" and path == "/v1/completions":
            try:
                payload = json.loads(body or b"null")
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._error(InvalidRequestError("Request body must be valid JSON."))
            return await self._gateway.handle_completion(headers.get("authorization"), payload)
        return self._error(NotFoundError())

    def _error(self, exc: Exception) -> GatewayResponse:
        request_id = uuid.uuid4().hex
        err = self._sanitizer.to_response(exc, request_id)
        return GatewayResponse(err.status, err.body, {**err.headers, "X-Request-Id": request_id})

    @staticmethod
    async def _write_response(writer: asyncio.StreamWriter, response: GatewayResponse) -> None:
        payload = json.dumps(response.body).encode("utf-8")
        reason = HTTPStatus(response.status).phrase
        head = [
            f"HTTP/1.1 {response.status} {reason}",
            "Content-Type: application/json",
            f"Content-Length: {len(payload)}",
            "Connection: close",
            "Cache-Control: no-store",
            "X-Content-Type-Options: nosniff",
        ]
        head += [f"{k}: {v}" for k, v in response.headers.items()]
        writer.write(("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + payload)
        await writer.drain()
