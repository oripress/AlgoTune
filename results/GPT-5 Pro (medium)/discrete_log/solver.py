from __future__ import annotations

from typing import Any, Dict, Tuple, List
import math
import random


def _sieve_primes(limit: int) -> List[int]:
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start : limit + 1 : step] = b"\x00" * ((limit - start) // step + 1)
    return [i for i, is_prime in enumerate(sieve) if is_prime]


# Precompute small primes once for fast trial division
_SMALL_PRIMES: List[int] = _sieve_primes(100000)


def _is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # Write n-1 as d * 2^s
    d = n - 1
    s = (d & -d).bit_length() - 1  # count factors of 2
    d >>= s
    # Deterministic bases for 64-bit integers
    # According to research, these bases are sufficient for n < 2^64
    # For larger n, we add more bases to reduce error probability drastically.
    if n < 2**64:
        bases = [2, 3, 5, 7, 11, 13, 17]
    else:
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in bases:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        skip_to_next_n = False
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                skip_to_next_n = True
                break
        if skip_to_next_n:
            continue
        return False
    return True


def _pollard_rho(n: int) -> int:
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        c = random.randrange(1, n - 1)
        f = lambda x: (x * x + c) % n
        x = random.randrange(2, n - 1)
        y = x
        d = 1
        # Use Brent's cycle detection to improve performance
        power = lam = 1
        while d == 1:
            if power == lam:
                x = y
                power <<= 1
                lam = 0
            y = f(y)
            lam += 1
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d
        # else retry with different parameters


def _factorize(n: int) -> Dict[int, int]:
    """Return prime factorization dict of n as {prime: exponent}."""
    factors: Dict[int, int] = {}
    if n <= 1:
        return factors
    # Trial division by small primes
    for p in _SMALL_PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            factors[p] = factors.get(p, 0) + e
        if n == 1:
            break
    if n == 1:
        return factors
    # If remainder is prime, done
    if _is_probable_prime(n):
        factors[n] = factors.get(n, 0) + 1
        return factors
    # Otherwise use Pollard Rho recursively
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if _is_probable_prime(m):
            factors[m] = factors.get(m, 0) + 1
            continue
        d = _pollard_rho(m)
        if d == m:
            # rare fallback: try again
            d = _pollard_rho(m)
            if d == m:
                # as a last resort, assume it's prime (extremely unlikely)
                factors[m] = factors.get(m, 0) + 1
                continue
        stack.append(d)
        stack.append(m // d)
    return factors


def _modinv(a: int, m: int) -> int:
    try:
        return pow(a, -1, m)
    except ValueError:
        # Fallback extended Euclid (in case older Python or non-invertible)
        g, x, _ = _egcd(a, m)
        if g != 1:
            raise ZeroDivisionError("Inverse does not exist")
        return x % m


def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = _egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def _bsgs(base: int, target: int, mod: int, order: int | None = None) -> int:
    """
    Baby-step Giant-step to solve base^x ≡ target (mod mod)
    If order is provided, x is found modulo 'order'.
    """
    if order is None:
        # In prime field multiplicative group, order divides mod-1.
        order = mod - 1
    order = int(order)
    if order == 0:
        return 0
    # Reduce base and target
    base %= mod
    target %= mod
    if target == 1:
        return 0
    m = int(math.isqrt(order) + 1)

    # Baby steps: base^j
    table: Dict[int, int] = {}
    cur = 1
    for j in range(m):
        # Store only first occurrence to keep minimal j
        if cur not in table:
            table[cur] = j
        cur = (cur * base) % mod

    # Compute base^{-m}
    base_m = pow(base, m, mod)
    inv_base_m = _modinv(base_m, mod)

    gamma = target
    for i in range(m + 1):
        if gamma in table:
            x = i * m + table[gamma]
            # Reduce modulo order
            return x % order
        gamma = (gamma * inv_base_m) % mod
    # If not found, raise error (should not happen for valid instances)
    raise ValueError("Discrete log not found via BSGS")


def _dlog_prime_power(p: int, g: int, h: int, q: int, e: int) -> int:
    """
    Solve x modulo q^e where g and h are in multiplicative group modulo prime p.
    Returns x in [0, q^e - 1] such that g^x ≡ h (mod p), assuming ord(g) is divisible by q^e.
    """
    n = p - 1
    mod_pi = q ** e
    # Reduce problem to subgroup of order q^e
    exponent = n // mod_pi
    g_i = pow(g, exponent, p)
    h_i = pow(h, exponent, p)
    # g_q has order q
    g_q = pow(g_i, q ** (e - 1), p)

    # If q is small, precompute table for g_q^k, else use BSGS with order q
    pre_table: Dict[int, int] | None = None
    SMALL_Q_THRESHOLD = 1 << 15  # 32768

    if q <= SMALL_Q_THRESHOLD:
        pre_table = {}
        cur = 1
        for k in range(q):
            if cur not in pre_table:
                pre_table[cur] = k
            cur = (cur * g_q) % p

    x = 0
    for j in range(e):
        # c = (h_i * g_i^{-x})^{q^{e-1-j}} mod p
        t = pow(g_i, x, p)
        inv_t = _modinv(t, p)
        c = (h_i * inv_t) % p
        c = pow(c, q ** (e - 1 - j), p)
        # Solve g_q^d = c mod p with d in [0, q-1]
        if pre_table is not None:
            d = pre_table.get(c)
            if d is None:
                # Should not happen if inputs are valid; fallback to BSGS
                d = _bsgs(g_q, c, p, order=q)
        else:
            d = _bsgs(g_q, c, p, order=q)
        x += d * (q ** j)
    return x % mod_pi


def _crt_pair(a1: int, m1: int, a2: int, m2: int) -> Tuple[int, int]:
    """Combine two congruences x ≡ a1 (mod m1), x ≡ a2 (mod m2). Returns (x, lcm)."""
    # m1 and m2 should be coprime here (prime powers of distinct primes)
    inv = _modinv(m1 % m2, m2)
    k = ((a2 - a1) % m2) * inv % m2
    x = a1 + m1 * k
    M = m1 * m2
    return x % M, M


def _discrete_log_prime_field(p: int, g: int, h: int) -> int:
    """
    Compute x such that g^x ≡ h (mod p), where p is prime and g is a generator.
    Uses Pohlig-Hellman with BSGS for large prime factors, falling back to BSGS if necessary.
    """
    if p == 2:
        # Only element is 1; h must be 1 and generator g == 1
        return 0
    n = p - 1
    g %= p
    h %= p
    if h == 1:
        return 0
    # Factor group order
    factors = _factorize(n)
    # If factoring failed (very unlikely), fallback to BSGS
    if not factors:
        return _bsgs(g, h, p, order=n)
    # If n is prime (only one factor with exponent 1), we can directly BSGS with that order
    if len(factors) == 1:
        (q, e), = factors.items()
        if e == 1:
            # Single large prime order, just BSGS with order q
            return _bsgs(pow(g, n // q, p), pow(h, n // q, p), p, order=q)

    # Pohlig-Hellman: solve modulo each prime power then combine via CRT
    x = 0
    modulus = 1
    for q, e in factors.items():
        xi = _dlog_prime_power(p, g, h, q, e)
        x, modulus = _crt_pair(x, modulus, xi, q ** e)
    return x % n


class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve discrete logarithm in a prime field: given prime p, generator g, find x with g^x ≡ h (mod p).
        """
        p = int(problem["p"])
        g = int(problem["g"]) % p
        h = int(problem["h"]) % p

        # Quick checks
        if p == 2:
            return {"x": 0}
        if h == 1:
            return {"x": 0}
        if g == 1:
            # 1^x = 1; So only solvable if h == 1 (already handled), else no solution; return 0 to be safe
            return {"x": 0}

        # Prefer Pohlig-Hellman with Pollard-Rho factorization; fallback to BSGS for robustness
        try:
            x = _discrete_log_prime_field(p, g, h)
        except Exception:
            # Robust fallback
            x = _bsgs(g, h, p, order=p - 1)
        return {"x": int(x)}