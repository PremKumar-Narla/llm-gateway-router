import asyncio
import json

from llm_gateway.api.http_server import GatewayHttpServer
from llm_gateway.providers.mock import Behavior, MockProvider
from tests.helpers import ACME_KEY, LEAK_MARKERS, TempDbTestCase


async def http(port: int, raw: bytes) -> tuple[int, dict[str, str], dict]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(raw)
    await writer.drain()
    data = await reader.read()
    writer.close()
    head, _, body = data.partition(b"\r\n\r\n")
    lines = head.decode().split("\r\n")
    status = int(lines[0].split(" ")[1])
    headers = {k.lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:])}
    return status, headers, json.loads(body)


def post(path: str, body: object, auth: str | None = None) -> bytes:
    payload = json.dumps(body).encode()
    auth_line = f"Authorization: {auth}\r\n" if auth else ""
    return (
        f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        f"{auth_line}Content-Length: {len(payload)}\r\n\r\n"
    ).encode() + payload


class HttpServerTest(TempDbTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        gw = await self.make_gateway([
            MockProvider("primary", "model-a", script=[Behavior.RATE_LIMIT]),
            MockProvider("secondary", "model-b"),
        ])
        self.server = GatewayHttpServer(gw, port=0)
        await self.server.start()
        self.addAsyncCleanup(self.server.close)

    async def test_completion_with_failover_over_http(self) -> None:
        status, headers, body = await http(
            self.server.port, post("/v1/completions", {"prompt": "hi"}, f"Bearer {ACME_KEY}")
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["fallback_used"])
        self.assertIn("x-request-id", headers)

    async def test_malformed_json_and_unknown_route(self) -> None:
        raw = (b"POST /v1/completions HTTP/1.1\r\nContent-Length: 5\r\n\r\n{oops")
        status, _, body = await http(self.server.port, raw)
        self.assertEqual((status, body["error"]["code"]), (400, "invalid_request"))

        status, _, body = await http(self.server.port, b"GET /admin HTTP/1.1\r\n\r\n")
        self.assertEqual((status, body["error"]["code"]), (404, "not_found"))

        status, _, body = await http(self.server.port, b"garbage\r\n\r\n")
        self.assertEqual(status, 400)
        for marker in LEAK_MARKERS:
            self.assertNotIn(marker, json.dumps(body))
