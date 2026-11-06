from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_gateway.clock import ManualClock
from llm_gateway.config import GatewayConfig, RateLimitConfig, RoutingConfig
from llm_gateway.gateway import LLMGateway
from llm_gateway.providers.base import ModelProvider

# Test-only key values (not real credentials).
ACME_KEY = "test-key-acme-000000000000"
GLOBEX_KEY = "test-key-globex-0000000000"

# Strings the mock providers put in upstream errors; must never reach clients.
LEAK_MARKERS = ("Traceback", "10.0.0.12", "primary", "secondary", "KeyError", "choices missing")


class TempDbTestCase(unittest.IsolatedAsyncioTestCase):
    """Gives each test its own on-disk SQLite file."""

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "gateway.db")
        self.clock = ManualClock()

    async def asyncTearDown(self) -> None:
        self._tmp.cleanup()

    async def make_gateway(
        self,
        providers: list[ModelProvider],
        *,
        tokens_per_window: int = 50_000,
        window_seconds: float = 60.0,
        attempt_timeout: float = 0.2,
    ) -> LLMGateway:
        config = GatewayConfig(
            database_path=self.db_path,
            rate_limit=RateLimitConfig(
                tokens_per_window=tokens_per_window, window_seconds=window_seconds
            ),
            routing=RoutingConfig(attempt_timeout_seconds=attempt_timeout),
        )
        gateway = await LLMGateway.create(config, providers, clock=self.clock)
        self.addAsyncCleanup(gateway.close)
        await gateway.register_api_key(ACME_KEY, tenant_id="acme")
        await gateway.register_api_key(GLOBEX_KEY, tenant_id="globex")
        return gateway
