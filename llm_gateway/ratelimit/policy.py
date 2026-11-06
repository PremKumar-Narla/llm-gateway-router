"""Pure sliding-window-log policy (no I/O, fully unit-testable).

Window semantics: a usage entry recorded at time `t` counts against the
budget while `t > now - window`, i.e. it expires at exactly `t + window`.
Eviction (`created_at <= window_start`) uses the same boundary, so the
admission check and the eviction can never disagree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SlidingWindowPolicy:
    limit: int
    window_seconds: float

    def window_start(self, now: float) -> float:
        return now - self.window_seconds

    def fits_at_all(self, requested: int) -> bool:
        """A request larger than the whole budget can never be admitted."""
        return requested <= self.limit

    def admits(self, used: int, requested: int) -> bool:
        return used + requested <= self.limit

    def retry_after(
        self,
        entries: Sequence[tuple[float, int]],
        used: int,
        requested: int,
        now: float,
    ) -> float:
        """Seconds until enough of the oldest entries expire to fit `requested`.

        `entries` are (created_at, tokens) for the live window, oldest first.
        """
        excess = used + requested - self.limit
        if excess <= 0:
            return 0.0
        freed = 0
        for created_at, tokens in entries:
            freed += tokens
            if freed >= excess:
                return max(0.0, created_at + self.window_seconds - now)
        # Only reachable if requested > limit, which callers reject earlier.
        return self.window_seconds
