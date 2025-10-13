from __future__ import annotations

import math
from typing import Any, Tuple, List

import sympy

def _sieve_primes(limit: int) -> List[int]:
    """Simple sieve of Eratosthenes to generate primes up to 'limit'."""
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    r = int(limit**0.5)
    for p in range(2, r + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start : limit + 1 : step] = b"\x00" * (((limit - start) // step) + 1)
    return [i for i, is_p in enumerate(sieve) if is_p]

# Precompute a modest list of small primes for quick trial division.
# Limit 10000 -> 1229 primes. Computed at import time (outside solve runtime).
_SMALL_PRIMES: List[int] = _sieve_primes(10000)

def _trial_division(n: int) -> Tuple[int, int] | None:
    """Try dividing by a set of small primes. Return (p, q) if found a factor, else None."""
    for p in _SMALL_PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            return p, n // p
    return None

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Factor a semiprime composite into p and q (p <= q).
        Strategy:
        - quick parity and perfect-square checks
        - small-prime trial division (cheap)
        - delegate to sympy.factorint (multiple=True) for robust, fast factorization
        """
        N = problem["composite"]
        N = int(N)

        # Even shortcut
        if (N & 1) == 0:
            p = 2
            q = N // 2
            if p > q:
                p, q = q, p
            return {"p": p, "q": q}

        # Perfect square shortcut (handles cases like p==q)
        r = math.isqrt(N)
        if r * r == N:
            return {"p": r, "q": r}

        # Small primes trial division (fast; rarely hits for balanced semiprimes, but cheap)
        td = _trial_division(N)
        if td is not None:
            p, q = td
            if p > q:
                p, q = q, p
            return {"p": p, "q": q}

        # Robust factorization via SymPy (uses multiple algorithms under the hood)
        # Use multiple=True to get a flat list of prime factors (slightly less overhead than dict)
        factors = sympy.factorint(N, multiple=True)

        if len(factors) != 2:
            # In principle, the benchmark generates semiprimes (possibly squares). Guard anyway.
            # If multiple factors or multiplicities, condense and verify.
            # Build dictionary to combine multiplicities
            fd = {}
            for f in factors:
                fd[f] = fd.get(f, 0) + 1
            expanded = [prime for prime, exp in fd.items() for _ in range(exp)]
            if len(expanded) != 2 or expanded[0] * expanded[1] != N:
                # As a last resort, fall back to default factorint dict
                fd = sympy.factorint(N)
                expanded = [prime for prime, exp in fd.items() for _ in range(exp)]
            factors = expanded

        if len(factors) != 2:
            # Should not happen for given problem generation; still handle gracefully
            # Try to create two factors p and q (product equals N)
            if factors:
                p = factors[0]
                q = N // int(p)
                if p > q:
                    p, q = q, p
                return {"p": int(p), "q": int(q)}
            # Fallback (unlikely)
            return {"p": 1, "q": N}

        p, q = factors
        if p > q:
            p, q = q, p
        return {"p": int(p), "q": int(q)}