from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict, **kwargs) -> NDArray:
        """
        Compute the cumulative integral of the 1D array using Simpson's rule for uniform spacing.

        Matches scipy.integrate.cumulative_simpson(y, dx=dx) behavior:
        - For even indices j = 2m, uses composite Simpson's rule on the first 2m intervals.
        - For odd indices j = 2m+1, adds a trapezoid for the last interval to the previous even result.
        - The first element is 0, returning an array of the same length as y.

        Parameters
        ----------
        problem: dict with keys:
            - "y": 1D array-like
            - "dx": float spacing between samples (uniform)

        Returns
        -------
        NDArray: cumulative integral array of the same length as y.
        """
        y = np.asarray(problem["y"], dtype=np.float64)
        dx = float(problem["dx"])
        n = y.size

        # Handle small sizes quickly
        if n == 0:
            return y  # empty
        res = np.empty(n, dtype=np.float64)
        res[0] = 0.0
        if n == 1:
            return res
        # Prepare even-index cumulative Simpson values for j = 2, 4, ..., j_even_max
        # Number of composite Simpson panels at j = 2m is m. The largest m satisfies 2m <= n-1 (largest even index).
        m_max = (n - 1) // 2  # number of even j values excluding j=0; m runs 1..m_max

        if m_max >= 1:
            # Cumulative sum of odd-indexed y: y[1], y[3], ..., y[2m-1] for m=1..m_max
            odd_vals = y[1 : 2 * m_max : 2]
            cum_odd = np.cumsum(odd_vals)  # length m_max

            # Cumulative sum of interior even-indexed y: y[2], y[4], ..., y[2m-2] for m=2..m_max
            even_interior_vals = y[2 : 2 * m_max : 2]  # length m_max - 1 (may be 0)
            if even_interior_vals.size > 0:
                cum_even_interior = np.cumsum(even_interior_vals)
            else:
                cum_even_interior = even_interior_vals  # empty

            # s_even_{m-1}: for m=1 it's 0, for m>=2 it's cum_even_interior[m-2]
            s_even_m_minus1 = np.empty(m_max, dtype=np.float64)
            s_even_m_minus1[0] = 0.0
            if m_max > 1:
                s_even_m_minus1[1:] = cum_even_interior

            # y at even endpoints y[2m] for m=1..m_max
            y_at_2m = y[2 : 2 * m_max + 1 : 2]  # length m_max
            y0 = y[0]

            # Composite Simpson for even indices
            I2m = (dx / 3.0) * (y0 + y_at_2m + 4.0 * cum_odd + 2.0 * s_even_m_minus1)

            # Populate results at even indices j = 2,4,...,2*m_max
            even_indices = np.arange(1, m_max + 1) * 2
            res[even_indices] = I2m

        # Odd indices via adding a trapezoid on top of previous even result
        odd_indices = np.arange(1, n, 2)
        res[odd_indices] = res[odd_indices - 1] + 0.5 * dx * (y[odd_indices - 1] + y[odd_indices])

        # If n is even, ensure last even index (n-2) already set; if n is odd, last even index n-1 handled above.
        # res[0] already set to 0. For n >= 2, all indices are now filled.

        return res