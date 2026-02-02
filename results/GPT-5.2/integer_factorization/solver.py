from __future__ import annotations

from dataclasses import dataclass
from math import gcd, isqrt
from typing import Any

# Small primes for quick trial division (cheap wins on small instances).
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def _xorshift64star(state: int) -> int:
    # Fast deterministic PRNG; state must be 64-bit.
    state ^= (state >> 12) & 0xFFFFFFFFFFFFFFFF
    state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
    state ^= (state >> 27) & 0xFFFFFFFFFFFFFFFF
    return (state * 2685821657736338717) & 0xFFFFFFFFFFFFFFFF

def _pollard_rho_brent(n: int, seed: int) -> int:
    """Return a non-trivial factor of odd composite n (Brent's cycle + batched gcd)."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3

    # Cheap perfect-square check.
    r = isqrt(n)
    if r * r == n:
        return r

    # Deterministic PRNG state derived from seed & n.
    st = (seed ^ (n & 0xFFFFFFFFFFFFFFFF) ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF

    # Brent parameters: m controls batch size (gcd frequency).
    m = 128

    while True:
        # Sample c in [1, n-1], y in [1, n-1]
        st = _xorshift64star(st)
        c = (st % (n - 1)) + 1
        st = _xorshift64star(st)
        y = (st % (n - 1)) + 1

        g = 1
        rpow = 1
        q = 1

        # f(z) = z^2 + c mod n
        while g == 1:
            x = y
            # advance y by rpow steps
            for _ in range(rpow):
                y = (y * y + c) % n

            k = 0
            while k < rpow and g == 1:
                ys = y
                lim = m if (rpow - k) > m else (rpow - k)
                # batch multiply differences
                for _ in range(lim):
                    y = (y * y + c) % n
                    diff = x - y
                    if diff < 0:
                        diff = -diff
                    q = (q * diff) % n
                g = gcd(q, n)
                k += lim

            rpow <<= 1

        if g == n:
            # backtrack to find actual factor
            while True:
                ys = (ys * ys + c) % n
                diff = x - ys
                if diff < 0:
                    diff = -diff
                g = gcd(diff, n)
                if g != 1:
                    break

        if 1 < g < n:
            return g
        # else: restart with new parameters

def _factor_semiprime(n: int) -> tuple[int, int]:
    """
    Factor n = p*q where p,q are primes.

    Important: for a true semiprime, *any* non-trivial factor returned by Pollard Rho
    is necessarily one of {p,q} (prime), so we avoid expensive primality testing.
    """
    # Quick handling for tiny n.
    if n <= 3:
        return 1, n

    # Very small composites: plain trial division beats rho overhead.
    if n < (1 << 16):
        if n % 2 == 0:
            return 2, n // 2
        r = isqrt(n)
        f = 3
        while f <= r:
            if n % f == 0:
                a, b = f, n // f
                return (a, b) if a < b else (b, a)
            f += 2
    # Fast small-prime trial division (very cheap for n=8/16-bit cases).
    for p in _SMALL_PRIMES:
        if n % p == 0:
            a, b = p, n // p
            return (a, b) if a < b else (b, a)

    # Perfect square (p==q) check.
    r = isqrt(n)
    if r * r == n:
        return r, r

    # Pollard rho (Brent)
    f = _pollard_rho_brent(n, seed=n ^ 0xD1B54A32D192ED03)
    g = n // f
    if f > g:
        f, g = g, f
    return f, g

@dataclass
class Solver:
    def solve(self, problem: dict[str, int], **kwargs: Any) -> Any:
        n = int(problem["composite"])
        p, q = _factor_semiprime(n)

        # Extremely defensive fallback (should never happen for benchmark generation).
        if p <= 1 or q <= 1 or p * q != n:
            import sympy  # type: ignore

            factors = [
                prime
                for prime, exp in sympy.factorint(sympy.Integer(n)).items()
                for _ in range(exp)
            ]
            factors.sort()
            if len(factors) != 2:
                raise ValueError(f"Expected 2 factors, got {len(factors)} for n={n}.")
            p, q = int(factors[0]), int(factors[1])

        if p > q:
            p, q = q, p
        return {"p": p, "q": q}