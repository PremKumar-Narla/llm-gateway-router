"""Run the gateway over HTTP with mock providers.

    GATEWAY_BOOTSTRAP_KEYS="acme=<api-key>" uv run python -m llm_gateway

Keys are read from the environment (never hard-coded) and stored hashed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from llm_gateway.api.http_server import GatewayHttpServer
from llm_gateway.config import GatewayConfig
from llm_gateway.gateway import LLMGateway
from llm_gateway.providers.mock import MockProvider


def _parse_bootstrap_keys(raw: str) -> list[tuple[str, str]]:
    pairs = []
    for item in filter(None, (p.strip() for p in raw.split(","))):
        tenant, sep, key = item.partition("=")
        if not sep or not tenant or not key:
            raise SystemExit("GATEWAY_BOOTSTRAP_KEYS must look like 'tenant=key,tenant2=key2'")
        pairs.append((tenant, key))
    return pairs


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = GatewayConfig(database_path=os.environ.get("GATEWAY_DB_PATH", "data/gateway.db"))

    providers = [
        # Flaky primary: ~30% of calls are 429s or hang past the 3s deadline.
        MockProvider("primary", "model-large", latency_seconds=0.2, failure_rate=0.3),
        MockProvider("secondary", "model-backup", latency_seconds=0.1),
    ]
    gateway = await LLMGateway.create(config, providers)

    keys = _parse_bootstrap_keys(os.environ.get("GATEWAY_BOOTSTRAP_KEYS", ""))
    if not keys:
        print("warning: GATEWAY_BOOTSTRAP_KEYS is empty; every request will be 401", file=sys.stderr)
    for tenant, key in keys:
        await gateway.register_api_key(key, tenant_id=tenant)

    server = GatewayHttpServer(
        gateway, host=os.environ.get("GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("GATEWAY_PORT", "8080")),
    )
    async with gateway:
        try:
            await server.serve_forever()
        finally:
            await server.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
