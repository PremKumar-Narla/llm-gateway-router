"""Injectable clocks so time-dependent logic is deterministic under test."""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Wall-clock seconds since the epoch (persisted to SQLite)."""
        ...


class SystemClock:
    def now(self) -> float:
        return time.time()


class ManualClock:
    """Test clock that only moves when told to."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds
