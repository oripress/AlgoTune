from __future__ import annotations

import math
import random
from typing import Any

class Solver:
    __slots__ = (
        "_factor_cache",
        "_group_cache",
        "_bsgs_cache",
        "_bsgs_cache_order",
        "_bsgs_cache_limit",
        "_small_primes",
    )

    def __init__(self) -> None:
        self._factor_cache: dict[int, dict[int, int]] = {}
        self._group_cache: dict[tuple[int, int], tuple[int, dict[int, int]]] = {}
        self._bsgs_cache: dict[tuple[int, int, int], tuple[int, dict[int, int], int]] = {}
        self._bsgs_cache_order: list[tuple[int, int, int]] = []
        self._bsgs_cache_limit = 8
        self._small_primes = self._make_small_primes(1000)

    def solve(self, problem, **kwargs) -> Any:
        p = int(problem["p"])
        g = int(problem["g"]) % p
        h = int(problem["h"]) % p

        try:
            if h == 1:
                return {"x": 0}
            if p == 2:
                return {"x": 0}
            if g == h:
                return {"x": 1}

            n = p - 1

            if n <= 1_000_000:
                try:
                    return {"x": self._bsgs(p, g, h, n, math.isqrt(n) + 1)}
                except Exception:
                    pass

            group_key = (p, g)
            group_data = self._group_cache.get(group_key)
            if group_data is None:
                full_factors = self._factor(n)

                order = n
                order_factors = full_factors.copy()
                for q in full_factors:
                    while order_factors.get(q, 0) > 0 and pow(g, order // q, p) == 1:
                        order //= q
                        new_e = order_factors[q] - 1
                        if new_e:
                            order_factors[q] = new_e
                        else:
                            del order_factors[q]

                self._group_cache[group_key] = (order, order_factors.copy())
                group_data = (order, order_factors)

            order, order_factors = group_data
            if order == 1:
                return {"x": 0}

            residues: list[tuple[int, int]] = []
            for q, e in order_factors.items():
                modulus = q**e
                residue = self._pohlig_hellman_prime_power(p, g, h, order, q, e)
                residues.append((residue, modulus))

            x = 0
            m = 1
            for residue, modulus in residues:
                t = ((residue - x) % modulus) * pow(m, -1, modulus) % modulus
                x += m * t
                m *= modulus
            return {"x": x % order}
        except Exception:
            from sympy.ntheory.residue_ntheory import discrete_log

            return {"x": int(discrete_log(p, h, g))}

    def _make_small_primes(self, limit: int) -> list[int]:
        sieve = bytearray(b"\x01") * (limit + 1)
        sieve[:2] = b"\x00\x00"
        end = int(limit**0.5) + 1
        for i in range(2, end):
            if sieve[i]:
                start = i * i
                sieve[start : limit + 1 : i] = b"\x00" * (((limit - start) // i) + 1)
        return [i for i in range(2, limit + 1) if sieve[i]]

    def _factor(self, n: int) -> dict[int, int]:
        cached = self._factor_cache.get(n)
        if cached is not None:
            return cached.copy()

        original = n
        factors: dict[int, int] = {}

        for p in self._small_primes:
            if p * p > n:
                break
            if n % p == 0:
                e = 1
                n //= p
                while n % p == 0:
                    n //= p
                    e += 1
                factors[p] = factors.get(p, 0) + e

        if n > 1:
            stack = [n]
            while stack:
                m = stack.pop()
                if m == 1:
                    continue
                if self._is_probable_prime(m):
                    factors[m] = factors.get(m, 0) + 1
                    continue
                d = self._pollard_brent(m)
                if d == m:
                    factors[m] = factors.get(m, 0) + 1
                else:
                    stack.append(d)
                    stack.append(m // d)

        result = dict(sorted(factors.items()))
        self._factor_cache[original] = result.copy()
        return result

    def _is_probable_prime(self, n: int) -> bool:
        if n < 2:
            return False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if n == p:
                return True
            if n % p == 0:
                return False

        d = n - 1
        s = 0
        while d & 1 == 0:
            s += 1
            d >>= 1

        if n < (1 << 64):
            bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
        else:
            bases = (2, 3, 5, 7, 11, 13, 17)

        for a in bases:
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def _pollard_brent(self, n: int) -> int:
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3

        randrange = random.randrange
        gcd = math.gcd

        while True:
            y = randrange(1, n - 1)
            c = randrange(1, n - 1)
            m = 128
            g = 1
            r = 1
            q = 1

            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n

                k = 0
                while k < r and g == 1:
                    ys = y
                    upper = min(m, r - k)
                    for _ in range(upper):
                        y = (y * y + c) % n
                        q = (q * abs(x - y)) % n
                    g = gcd(q, n)
                    k += m
                r <<= 1

            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)

            if g != n:
                return g

    def _pohlig_hellman_prime_power(
        self,
        p: int,
        g: int,
        h: int,
        order: int,
        q: int,
        e: int,
    ) -> int:
        exp = order // q
        gamma = pow(g, exp, p)

        if e == 1:
            return self._discrete_log_prime_order(p, gamma, pow(h, exp, p), q)

        x = 0
        qk = 1
        g_inv = pow(g, -1, p)
        g_inv_x = 1
        g_inv_qk = g_inv
        dlog = self._discrete_log_prime_order

        for i in range(e):
            d = pow((h * g_inv_x) % p, exp, p)
            a = dlog(p, gamma, d, q)
            if a:
                x += a * qk
                g_inv_x = (g_inv_x * pow(g_inv_qk, a, p)) % p
            qk *= q
            exp //= q
            if i + 1 < e:
                g_inv_qk = pow(g_inv_qk, q, p)

        return x

    def _discrete_log_prime_order(self, p: int, base: int, target: int, order: int) -> int:
        if target == 1:
            return 0
        if target == base:
            return 1

        if order <= 256:
            cur = 1
            for i in range(order):
                if cur == target:
                    return i
                cur = (cur * base) % p
            raise ValueError("discrete log not found")

        m = math.isqrt(order) + 1
        if m <= 1_000_000:
            return self._bsgs(p, base, target, order, m)
        return self._pollard_rho_dlog(p, base, target, order)

    def _bsgs(self, p: int, base: int, target: int, order: int, m: int | None = None) -> int:
        if m is None:
            m = math.isqrt(order) + 1

        use_cache = m <= 400_000
        key = (p, base, order)
        cached = self._bsgs_cache.get(key) if use_cache else None

        if cached is None:
            table: dict[int, int] = {}
            cur = 1
            mod = p
            mul = base
            for j in range(m):
                table[cur] = j
                cur = (cur * mul) % mod
            factor = 1 if m % order == 0 else pow(base, order - (m % order), p)
            if use_cache:
                cached = (m, table, factor)
                self._bsgs_cache[key] = cached
                self._bsgs_cache_order.append(key)
                if len(self._bsgs_cache_order) > self._bsgs_cache_limit:
                    old = self._bsgs_cache_order.pop(0)
                    self._bsgs_cache.pop(old, None)
        else:
            m, table, factor = cached

        gamma = target
        get = table.get
        mod = p
        step = factor
        ord_ = order
        jump = m
        for i in range(m + 1):
            j = get(gamma)
            if j is not None:
                x = i * jump + j
                if x < ord_:
                    return x
            gamma = (gamma * step) % mod

        raise ValueError("discrete log not found")

    def _pollard_rho_dlog(self, p: int, alpha: int, beta: int, order: int) -> int:
        if order == 2:
            return 1 if beta == alpha else 0

        randrange = random.randrange
        for _ in range(32):
            a = randrange(order)
            b = randrange(order)
            x = (pow(alpha, a, p) * pow(beta, b, p)) % p
            A, B, X = a, b, x

            while True:
                r = x & 15
                if r < 5:
                    x = (x * beta) % p
                    b = (b + 1) % order
                elif r < 10:
                    x = (x * x) % p
                    a = (a << 1) % order
                    b = (b << 1) % order
                else:
                    x = (x * alpha) % p
                    a = (a + 1) % order

                for _ in range(2):
                    r = X & 15
                    if r < 5:
                        X = (X * beta) % p
                        B = (B + 1) % order
                    elif r < 10:
                        X = (X * X) % p
                        A = (A << 1) % order
                        B = (B << 1) % order
                    else:
                        X = (X * alpha) % p
                        A = (A + 1) % order

                if x == X:
                    denom = (B - b) % order
                    if denom == 0:
                        break
                    return ((a - A) % order) * pow(denom, -1, order) % order

        raise ValueError("pollard rho discrete log failed")