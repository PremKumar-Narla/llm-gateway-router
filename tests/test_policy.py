import unittest

from llm_gateway.ratelimit.policy import SlidingWindowPolicy


class SlidingWindowPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = SlidingWindowPolicy(limit=100, window_seconds=60)

    def test_admits_up_to_exact_limit(self) -> None:
        self.assertTrue(self.policy.admits(used=60, requested=40))
        self.assertFalse(self.policy.admits(used=60, requested=41))

    def test_rejects_request_larger_than_budget(self) -> None:
        self.assertFalse(self.policy.fits_at_all(101))
        self.assertTrue(self.policy.fits_at_all(100))

    def test_retry_after_waits_for_oldest_entries_needed(self) -> None:
        now = 1_000.0
        # 3 entries in window: t=950 (30), t=970 (30), t=990 (40) -> used 100
        entries = [(950.0, 30), (970.0, 30), (990.0, 40)]
        # Need 50 -> must free 50 -> first two entries (60) -> wait until 970+60
        self.assertAlmostEqual(self.policy.retry_after(entries, 100, 50, now), 30.0)
        # Need 10 -> freeing the first entry is enough -> 950+60-1000
        self.assertAlmostEqual(self.policy.retry_after(entries, 100, 10, now), 10.0)

    def test_retry_after_zero_when_it_fits(self) -> None:
        self.assertEqual(self.policy.retry_after([], 10, 10, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
