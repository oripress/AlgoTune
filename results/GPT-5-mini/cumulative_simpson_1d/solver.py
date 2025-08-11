import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: Any, **kwargs) -> Any:
        """
        Compute the cumulative integral of a 1D array using composite Simpson's rule
        for uniformly spaced samples (dx provided). Returns an array of length n-1
        matching scipy.integrate.cumulative_simpson(y, dx=dx).
        """
        y = problem["y"]
        dx = float(problem["dx"])

        ya = np.asarray(y, dtype=np.float64)
        if ya.ndim != 1:
            ya = ya.ravel()
        n = ya.size

        if n <= 1:
            return np.empty(0, dtype=np.float64)

        out = np.empty(n - 1, dtype=np.float64)

        # First interval via trapezoid
        out[0] = 0.5 * dx * (ya[0] + ya[1])

        # Number of Simpson even endpoints: k = 2,4,...,2*M
        M = (n - 1) // 2

        if M > 0:
            # y at even endpoints: y[2], y[4], ..., y[2*M]
            y_evens = ya[2:2 * M + 1:2]

            # y at odd positions used in Simpson sums: y[1], y[3], ..., y[2*M-1]
            y_odds = ya[1:2 * M + 1:2]
            odd_cumsum = np.cumsum(y_odds, dtype=np.float64)

            # interior even samples for Simpson sums: y[2], y[4], ..., y[2*M-2]
            even_interior = ya[2:2 * M:2]
            if even_interior.size > 0:
                even_cumsum = np.cumsum(even_interior, dtype=np.float64)
                even_sum = np.empty(M, dtype=np.float64)
                even_sum[0] = 0.0
                even_sum[1:] = even_cumsum
            else:
                even_sum = np.zeros(M, dtype=np.float64)

            coeff = dx / 3.0
            even_integrals = coeff * (ya[0] + y_evens + 4.0 * odd_cumsum + 2.0 * even_sum)
            out[1:2 * M + 1:2] = even_integrals

        # For odd positions (k odd: 3,5,...), use previous even Simpson + trapezoid
        odd_idx = np.arange(2, n - 1, 2, dtype=np.intp)
        if odd_idx.size > 0:
            out[odd_idx] = out[odd_idx - 1] + 0.5 * dx * (ya[odd_idx] + ya[odd_idx + 1])

        return out