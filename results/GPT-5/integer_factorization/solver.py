from __future__ import annotations

from typing import Any
import sympy as sp

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Factor a composite integer that is the product of two primes p and q.
        Returns {"p": p, "q": q} with p < q.
        """
        if not isinstance(problem, dict) or "composite" not in problem:
            raise ValueError("Problem must be a dict containing key 'composite'.")
        n = problem["composite"]
        if not isinstance(n, int):
            try:
                n = int(n)
            except Exception as exc:
                raise ValueError(f"Invalid composite value: {problem.get('composite')}") from exc

        # Factor using sympy
        fac = sp.factorint(n)  # dict: {prime: exponent}
        factors: list[int] = []
        for prime, exp in fac.items():
            factors.extend([int(prime)] * int(exp))

        if len(factors) != 2:
            raise ValueError(f"Expected 2 factors, got {len(factors)}")

        p, q = sorted((int(factors[0]), int(factors[1])))
        return {"p": p, "q": q}