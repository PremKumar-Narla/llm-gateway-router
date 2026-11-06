"""End-to-end walkthrough of the gateway's behaviour (no network needed).

    uv run python demo.py
"""

from __future__ import annotations

import asyncio
import json
import secrets
import tempfile
from pathlib import Path

from llm_gateway import GatewayConfig, LLMGateway, RateLimitConfig, RoutingConfig
from llm_gateway.providers.mock import Behavior, MockProvider


def show(title: str, res) -> None:
    print(f"\n== {title} -> HTTP {res.status}")
    print(json.dumps(res.body, indent=2))


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = GatewayConfig(
            database_path=str(Path(tmp) / "gateway.db"),
            rate_limit=RateLimitConfig(tokens_per_window=1_000, window_seconds=60),
            routing=RoutingConfig(attempt_timeout_seconds=3.0),
        )
        primary = MockProvider(
            "primary", "model-large", latency_seconds=0.05,
            script=[Behavior.SUCCESS, Behavior.RATE_LIMIT, Behavior.HANG],
        )
        secondary = MockProvider("secondary", "model-backup", latency_seconds=0.05)

        async with await LLMGateway.create(config, [primary, secondary]) as gw:
            api_key = secrets.token_urlsafe(24)  # generated per run, never printed
            await gw.register_api_key(api_key, tenant_id="acme")
            auth = f"Bearer {api_key}"
            body = {"prompt": "Explain sliding window rate limiting", "max_tokens": 64}

            show("1. primary healthy", await gw.handle_completion(auth, body))
            show("2. primary returns 429 -> failover", await gw.handle_completion(auth, body))
            print("\n(3. primary hangs; waiting for the 3s deadline...)")
            show("3. primary times out -> failover", await gw.handle_completion(auth, body))
            show("4. bad API key", await gw.handle_completion("Bearer nope-nope-nope-nope", body))
            show("5. fits the budget, but not what is left of it",
                 await gw.handle_completion(auth, {"prompt": "x" * 1_600, "max_tokens": 580}))
            show("6. larger than the whole budget",
                 await gw.handle_completion(auth, {"prompt": "x" * 2_000, "max_tokens": 600}))


if __name__ == "__main__":
    asyncio.run(main())
