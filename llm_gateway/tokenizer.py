"""Token estimation used to *reserve* budget before the upstream call.

The estimate is reconciled with the provider's reported usage afterwards, so
a rough heuristic is acceptable here. Swap in a real tokenizer by
implementing `TokenEstimator`.
"""

from __future__ import annotations

import math
from typing import Protocol


class TokenEstimator(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenEstimator:
    """~4 characters per token, rounded up (errs on over-reserving)."""

    def __init__(self, chars_per_token: float = 4.0) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self._chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self._chars_per_token))
