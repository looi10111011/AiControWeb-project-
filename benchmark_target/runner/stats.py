"""Wilson score interval (spec 32) — diagnostic only, never overrides the deterministic
STABLE_PASS/STABLE_FAIL/FLAKY/INCONCLUSIVE classification (spec 30).
"""

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval by default (z=1.96). Returns (low, high) in [0, 1]."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    low = (center - margin) / denom
    high = (center + margin) / denom
    return (max(0.0, low), min(1.0, high))
