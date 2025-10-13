from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: Tuple[int, Tuple[float, float]], **kwargs) -> Any:
        """
        Matches scipy.signal.firls for a two-band design:
        Given (n, (f1, f2)), constructs an odd-length filter with N = 2*n + 1 taps,
        and designs least-squares FIR with bands [0, f1] (gain 1) and [f2, 1] (gain 0).
        """
        n, edges = problem
        N = 2 * int(n) + 1  # must be odd
        f1, f2 = tuple(edges)
        bands = (0.0, f1, f2, 1.0)
        return signal.firls(N, bands, [1, 1, 0, 0])