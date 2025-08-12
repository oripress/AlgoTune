import sympy
from typing import Any, Dict

class Solver:
    """
    Simple solver delegating factorization to sympy.factorint.
    Assumes the input composite is the product of two primes.
    """

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, int]:
        composite_val = problem.get("composite")
        if composite_val is None:
            raise ValueError("Problem must contain 'composite'")

        try:
            n = int(composite_val)
        except Exception as e:
            raise ValueError(f"Invalid composite value: {e}")

        if n <= 1:
            raise ValueError("Composite must be an integer > 1")

        # Use sympy's factorint which is highly optimized for typical sizes here
        fac = sympy.factorint(n)
        factors = []
        for prime, exp in fac.items():
            factors.extend([int(prime)] * int(exp))

        if len(factors) == 0:
            raise ValueError("Failed to factor the composite number")

        factors.sort()
        # If there's only one factor (shouldn't happen for composite of two primes), try to derive the other
        if len(factors) == 1:
            p = factors[0]
            if p != 0 and n % p == 0:
                q = n // p
                p, q = sorted([int(p), int(q)])
                return {"p": p, "q": q}
            raise ValueError("Unexpected factorization result")

        # If more than two factors (rare for this task), combine smallest factors until two remain
        while len(factors) > 2:
            a = factors.pop(0)
            b = factors.pop(0)
            factors.append(a * b)
            factors.sort()

        p, q = sorted([int(factors[0]), int(factors[1])])
        return {"p": p, "q": q}