from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy.special import wright_bessel

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Compute integrals of Wright's Bessel function using an exact antiderivative identity.

        Wright's Bessel function:
            Phi(a, b; x) = sum_{k=0}^\infty x^k / (k! * Gamma(a k + b))

        Identity:
            d/dx Phi(a, b - a; x) = Phi(a, b; x)

        Hence:
            ∫_L^U Phi(a, b; x) dx = Phi(a, b - a; U) - Phi(a, b - a; L)

        Parameters:
            problem: dict with keys "a", "b", "lower", "upper" (lists of equal length)

        Returns:
            dict with key "result": list of integrals for each (a, b, lower, upper)
        """
        a_list: List[float] = problem["a"]
        b_list: List[float] = problem["b"]
        lower_list: List[float] = problem["lower"]
        upper_list: List[float] = problem["upper"]

        a = np.asarray(a_list, dtype=np.float64)
        b = np.asarray(b_list, dtype=np.float64)
        L = np.asarray(lower_list, dtype=np.float64)
        U = np.asarray(upper_list, dtype=np.float64)

        if not (a.shape == b.shape == L.shape == U.shape):
            raise ValueError("Input arrays must have the same shape")

        bp = b - a  # shifted parameter for antiderivative
        res = wright_bessel(U, a, bp) - wright_bessel(L, a, bp)

        # Ensure finite results
        if not np.all(np.isfinite(res)):
            raise FloatingPointError("Non-finite result encountered in integration")

        return {"result": res.tolist()}