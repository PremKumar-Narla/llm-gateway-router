import asyncio
import time
import unittest

from llm_gateway.config import RoutingConfig
from llm_gateway.models import CompletionRequest, ProviderCompletion
from llm_gateway.providers.base import ModelProvider
from llm_gateway.providers.errors import FailureKind
from llm_gateway.providers.mock import Behavior, MockProvider
from llm_gateway.routing.router import FallbackRouter, RoutingFailedError

REQ = CompletionRequest(prompt="hello", max_tokens=16)


def router(*providers: ModelProvider, timeout: float = 0.1) -> FallbackRouter:
    return FallbackRouter(providers, RoutingConfig(attempt_timeout_seconds=timeout))


class FallbackRouterTest(unittest.IsolatedAsyncioTestCase):
    async def test_primary_success_does_not_touch_secondary(self) -> None:
        primary, secondary = MockProvider("p", "m1"), MockProvider("s", "m2")
        result = await router(primary, secondary).route(REQ)
        self.assertEqual(result.provider, "p")
        self.assertFalse(result.fallback_used)
        self.assertEqual(secondary.calls, 0)

    async def test_429_fails_over_to_secondary(self) -> None:
        primary = MockProvider("p", "m1", script=[Behavior.RATE_LIMIT])
        secondary = MockProvider("s", "m2")
        result = await router(primary, secondary).route(REQ)
        self.assertEqual(result.provider, "s")
        self.assertTrue(result.fallback_used)
        self.assertEqual([a.outcome for a in result.attempts], ["rate_limited", "success"])

    async def test_timeout_cancels_primary_before_secondary_starts(self) -> None:
        primary = MockProvider("p", "m1", script=[Behavior.HANG])
        secondary_started_after_cancel = []

        class Secondary(MockProvider):
            async def complete(self, request):
                secondary_started_after_cancel.append(primary.cancelled == 1)
                return await super().complete(request)

        started = time.perf_counter()
        result = await router(primary, Secondary("s", "m2"), timeout=0.1).route(REQ)
        elapsed = time.perf_counter() - started

        self.assertEqual(result.provider, "s")
        self.assertEqual(result.attempts[0].outcome, "timeout")
        self.assertEqual(secondary_started_after_cancel, [True])
        self.assertGreaterEqual(elapsed, 0.1)
        self.assertLess(elapsed, 0.5)
        # No stray tasks left behind by the timed-out attempt.
        self.assertEqual(len(asyncio.all_tasks()), 1)

    async def test_slow_but_within_deadline_is_not_a_timeout(self) -> None:
        primary = MockProvider("p", "m1", latency_seconds=0.05)
        result = await router(primary, MockProvider("s", "m2"), timeout=0.2).route(REQ)
        self.assertEqual(result.provider, "p")

    async def test_non_failover_error_fails_fast(self) -> None:
        primary = MockProvider("p", "m1", script=[Behavior.BAD_REQUEST])
        secondary = MockProvider("s", "m2")
        with self.assertRaises(RoutingFailedError) as ctx:
            await router(primary, secondary).route(REQ)
        self.assertEqual(ctx.exception.final_kind, FailureKind.BAD_REQUEST)
        self.assertEqual(secondary.calls, 0)

    async def test_all_providers_exhausted(self) -> None:
        primary = MockProvider("p", "m1", script=[Behavior.RATE_LIMIT])
        secondary = MockProvider("s", "m2", script=[Behavior.HANG])
        with self.assertRaises(RoutingFailedError) as ctx:
            await router(primary, secondary).route(REQ)
        self.assertEqual(ctx.exception.final_kind, FailureKind.TIMEOUT)
        self.assertEqual(len(ctx.exception.attempts), 2)

    async def test_adapter_crash_is_classified_not_propagated(self) -> None:
        primary = MockProvider("p", "m1", script=[Behavior.CRASH])
        with self.assertRaises(RoutingFailedError) as ctx:
            await router(primary, MockProvider("s", "m2")).route(REQ)
        self.assertEqual(ctx.exception.final_kind, FailureKind.UNKNOWN)

    async def test_external_cancellation_is_not_treated_as_failover(self) -> None:
        primary = MockProvider("p", "m1", script=[Behavior.HANG])
        secondary = MockProvider("s", "m2")
        task = asyncio.create_task(router(primary, secondary, timeout=5).route(REQ))
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(primary.cancelled, 1)
        self.assertEqual(secondary.calls, 0)

    async def test_result_racing_the_deadline_is_decided_once(self) -> None:
        """Latency ~= timeout: outcome is either success or a clean failover,
        never both providers' results and never an unhandled error."""

        class Racer(ModelProvider):
            async def complete(self, request):
                await asyncio.sleep(0.05)
                return ProviderCompletion("x", self.model, 1, 1)

        for _ in range(20):
            secondary = MockProvider("s", "m2")
            result = await router(Racer("p", "m1"), secondary, timeout=0.05).route(REQ)
            self.assertIn(result.provider, {"p", "s"})
            self.assertEqual(secondary.calls, 1 if result.provider == "s" else 0)


if __name__ == "__main__":
    unittest.main()
