import asyncio
import json

from llm_gateway.providers.mock import Behavior, MockProvider
from tests.helpers import ACME_KEY, GLOBEX_KEY, LEAK_MARKERS, TempDbTestCase

AUTH = f"Bearer {ACME_KEY}"
BODY = {"prompt": "Summarise the quarterly report", "max_tokens": 100}


class GatewayTest(TempDbTestCase):
    def assert_no_leak(self, body: dict) -> None:
        text = json.dumps(body)
        for marker in LEAK_MARKERS:
            self.assertNotIn(marker, text)

    def assert_error_shape(self, body: dict, code: str) -> None:
        self.assertEqual(set(body), {"error"})
        self.assertEqual(body["error"]["code"], code)
        self.assertIn("message", body["error"])
        self.assertIn("request_id", body["error"])

    async def test_success_commits_actual_usage(self) -> None:
        gw = await self.make_gateway([MockProvider("primary", "model-a")])
        res = await gw.handle_completion(AUTH, BODY)
        self.assertEqual(res.status, 200)
        self.assertFalse(res.body["fallback_used"])
        principal = await gw.authenticate(AUTH)
        # Estimate was prompt(8) + max_tokens(100); committed is actual usage.
        self.assertEqual(await gw.limiter.usage(principal), res.body["usage"]["total_tokens"])
        self.assertIn("X-RateLimit-Remaining", res.headers)
        self.assert_no_leak(res.body)

    async def test_fallback_on_429_is_transparent_and_audited(self) -> None:
        gw = await self.make_gateway([
            MockProvider("primary", "model-a", script=[Behavior.RATE_LIMIT]),
            MockProvider("secondary", "model-b"),
        ])
        res = await gw.handle_completion(AUTH, BODY)
        self.assertEqual(res.status, 200)
        self.assertTrue(res.body["fallback_used"])
        self.assertEqual(res.body["model"], "model-b")
        audit = await gw.audit.get(res.headers["X-Request-Id"])
        self.assertEqual(audit["provider"], "secondary")
        self.assertEqual([a["outcome"] for a in audit["attempts"]], ["rate_limited", "success"])

    async def test_fallback_on_timeout(self) -> None:
        gw = await self.make_gateway(
            [MockProvider("primary", "model-a", script=[Behavior.HANG]),
             MockProvider("secondary", "model-b")],
            attempt_timeout=0.1,
        )
        res = await gw.handle_completion(AUTH, BODY)
        self.assertEqual(res.status, 200)
        self.assertEqual(res.body["model"], "model-b")

    async def test_all_providers_down_returns_sanitized_error_and_refunds(self) -> None:
        gw = await self.make_gateway([
            MockProvider("primary", "model-a", default=Behavior.RATE_LIMIT),
            MockProvider("secondary", "model-b", default=Behavior.RATE_LIMIT),
        ])
        res = await gw.handle_completion(AUTH, BODY)
        self.assertEqual(res.status, 503)
        self.assert_error_shape(res.body, "upstream_unavailable")
        self.assert_no_leak(res.body)
        principal = await gw.authenticate(AUTH)
        self.assertEqual(await gw.limiter.usage(principal), 0)  # reservation released

    async def test_timeout_on_every_provider_maps_to_504(self) -> None:
        gw = await self.make_gateway(
            [MockProvider("primary", "a", default=Behavior.HANG),
             MockProvider("secondary", "b", default=Behavior.HANG)],
            attempt_timeout=0.05,
        )
        res = await gw.handle_completion(AUTH, BODY)
        self.assertEqual(res.status, 504)
        self.assert_error_shape(res.body, "upstream_timeout")

    async def test_upstream_500_and_adapter_crash_do_not_leak(self) -> None:
        for behavior, status in ((Behavior.SERVER_ERROR, 502), (Behavior.CRASH, 502),
                                 (Behavior.BAD_REQUEST, 400)):
            with self.subTest(behavior=behavior):
                await self.asyncTearDown()
                await self.asyncSetUp()
                gw = await self.make_gateway([MockProvider("primary", "a", default=behavior)])
                with self.assertLogs("llm_gateway", level="WARNING"):  # detail goes to logs
                    res = await gw.handle_completion(AUTH, BODY)
                self.assertEqual(res.status, status)
                self.assert_no_leak(res.body)

    async def test_rate_limit_returns_429_with_retry_after(self) -> None:
        gw = await self.make_gateway([MockProvider("primary", "a")], tokens_per_window=250)
        first = await gw.handle_completion(AUTH, {"prompt": "x" * 400, "max_tokens": 100})
        self.assertEqual(first.status, 200)  # 200 reserved, ~124 committed
        second = await gw.handle_completion(AUTH, {"prompt": "x" * 400, "max_tokens": 100})
        self.assertEqual(second.status, 429)
        self.assert_error_shape(second.body, "rate_limit_exceeded")
        self.assertEqual(second.body["error"]["retry_after_seconds"], 60.0)
        self.assertEqual(second.headers["Retry-After"], "60")
        # Another tenant's key is unaffected.
        other = await gw.handle_completion(f"Bearer {GLOBEX_KEY}", BODY)
        self.assertEqual(other.status, 200)
        # After the window slides, acme is admitted again.
        self.clock.advance(60)
        self.assertEqual((await gw.handle_completion(AUTH, BODY)).status, 200)

    async def test_oversized_request_is_413(self) -> None:
        gw = await self.make_gateway([MockProvider("primary", "a")], tokens_per_window=50)
        res = await gw.handle_completion(AUTH, {"prompt": "hi", "max_tokens": 100})
        self.assertEqual(res.status, 413)
        self.assert_error_shape(res.body, "request_too_large")

    async def test_auth_and_validation_errors(self) -> None:
        gw = await self.make_gateway([MockProvider("primary", "a")])
        cases = [
            (None, BODY, 401, "unauthenticated"),
            ("Bearer test-key-unknown-00000000", BODY, 401, "unauthenticated"),
            ("Basic abc", BODY, 401, "unauthenticated"),
            (AUTH, {"prompt": ""}, 400, "invalid_request"),
            (AUTH, {"prompt": "hi", "max_tokens": True}, 400, "invalid_request"),
            (AUTH, {"prompt": "hi", "max_tokens": 10**9}, 400, "invalid_request"),
            (AUTH, ["not", "an", "object"], 400, "invalid_request"),
        ]
        for auth, body, status, code in cases:
            with self.subTest(auth=auth, body=body):
                res = await gw.handle_completion(auth, body)
                self.assertEqual(res.status, status)
                self.assert_error_shape(res.body, code)

    async def test_client_cancellation_releases_reservation(self) -> None:
        gw = await self.make_gateway(
            [MockProvider("primary", "a", default=Behavior.HANG)], attempt_timeout=5
        )
        task = asyncio.create_task(gw.handle_completion(AUTH, BODY))
        await asyncio.sleep(0.05)
        principal = await gw.authenticate(AUTH)
        self.assertGreater(await gw.limiter.usage(principal), 0)  # reserved in-flight
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(await gw.limiter.usage(principal), 0)

    async def test_concurrent_requests_respect_budget(self) -> None:
        gw = await self.make_gateway(
            [MockProvider("primary", "a", latency_seconds=0.02)], tokens_per_window=1_000
        )
        results = await asyncio.gather(
            *(gw.handle_completion(AUTH, {"prompt": "x" * 40, "max_tokens": 90})
              for _ in range(30))
        )
        statuses = [r.status for r in results]
        # Each reserves 100 up front, so at most 10 run concurrently.
        self.assertEqual(statuses.count(200), 10)
        self.assertEqual(statuses.count(429), 20)
