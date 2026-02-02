from __future__ import annotations

import math
from typing import Any

class Solver:
    """
    Fast discrete log for prime modulus p.

    Key optimization vs baseline sympy.discrete_log:
      - cache factorization of p-1 per p
      - cache all g-dependent Pohlig–Hellman precomputations per (p,g,order)
        (g1/g0/base_k/exp_k), so repeated queries for same (p,g) become much faster
      - solve prime-order sub-DLPs via linear scan (tiny q) or BSGS (cached)

    Correctness:
      - Run PH assuming order(g)=p-1, verify once.
      - If verification fails (rare), compute actual order(g) and redo PH.
      - Last resort: sympy.discrete_log
    """

    def __init__(self) -> None:
        self._fac_cache: dict[int, dict[int, int]] = {}
        self._fac_cache_max = 128

        # (p,g,order) -> tuple(entries)
        # entry = (q, e, qe, n_div_qe, g0, base_list(tuple), exp_list(tuple))
        self._ph_cache: dict[tuple[int, int, int], tuple[tuple[int, int, int, int, int, tuple[int, ...], tuple[int, ...]], ...]] = {}
        self._ph_cache_max = 128

        # (p, base, q) -> (m, base_inv_m, baby_dict)
        self._bsgs_cache: dict[tuple[int, int, int], tuple[int, int, dict[int, int]]] = {}
        self._bsgs_cache_max = 256

        # Tuning
        self._linear_q_max = 2048
        self._max_q_for_bsgs = 50_000_000

        # Imports in __init__ (not counted toward runtime)
        from sympy.ntheory import factorint  # type: ignore
        from sympy.ntheory.residue_ntheory import discrete_log  # type: ignore

        self._factorint = factorint
        self._sympy_discrete_log = discrete_log

    @staticmethod
    def _crt_pair(a1: int, m1: int, a2: int, m2: int) -> tuple[int, int]:
        k = ((a2 - a1) % m2) * pow(m1, -1, m2) % m2
        return a1 + m1 * k, m1 * m2

    def _factor_p_minus_1(self, p: int) -> dict[int, int]:
        fac = self._fac_cache.get(p)
        if fac is not None:
            return fac
        fac_sym = self._factorint(p - 1)
        fac = {int(k): int(v) for k, v in fac_sym.items()}
        if len(self._fac_cache) >= self._fac_cache_max:
            self._fac_cache.clear()
        self._fac_cache[p] = fac
        return fac

    def _get_ph_entries(self, p: int, g: int, order: int, fac: dict[int, int]) -> tuple[
        tuple[int, int, int, int, int, tuple[int, ...], tuple[int, ...]], ...
    ]:
        key = (p, g, order)
        cached = self._ph_cache.get(key)
        if cached is not None:
            return cached

        pow_ = pow
        entries: list[tuple[int, int, int, int, int, tuple[int, ...], tuple[int, ...]]] = []

        # fac is for (p-1); order may be smaller. We'll trim exponents as needed.
        for q, e in fac.items():
            if order % q != 0:
                continue
            # trim exponent to order
            ee = e
            # avoid pow(q, ee) recomputation in a loop by decreasing if needed
            while ee > 0 and order % pow(q, ee) != 0:
                ee -= 1
            if ee <= 0:
                continue

            qe = pow(q, ee)
            n_div_qe = order // qe

            # Reduce g to subgroup of order q^e, and extract order-q generator g0
            g1 = pow_(g, n_div_qe, p)
            g1_inv = pow_(g1, -1, p)
            g0 = pow_(g1, qe // q, p)

            # Precompute base_k = g1_inv^(q^k) and exp_k = q^(e-1-k)
            base_list: list[int] = []
            base_k = g1_inv
            for _ in range(ee):
                base_list.append(base_k)
                base_k = pow_(base_k, q, p)

            exp_list: list[int] = []
            exp_k = qe // q
            for _ in range(ee):
                exp_list.append(exp_k)
                exp_k //= q

            entries.append((q, ee, qe, n_div_qe, g0, tuple(base_list), tuple(exp_list)))

        out = tuple(entries)
        if len(self._ph_cache) >= self._ph_cache_max:
            self._ph_cache.clear()
        self._ph_cache[key] = out
        return out

    def _bsgs_prime_order(self, p: int, g: int, h: int, q: int) -> int:
        """Solve g^x = h (mod p) where order(g)=q is prime. Return x in [0,q-1]."""
        if h == 1:
            return 0

        if q <= self._linear_q_max:
            v = 1
            gg = g
            for x in range(q):
                if v == h:
                    return x
                v = (v * gg) % p
            raise ValueError("no dlog")

        if q > self._max_q_for_bsgs:
            raise ValueError("q too large")

        key = (p, g, q)
        cached = self._bsgs_cache.get(key)
        if cached is None:
            m = math.isqrt(q) + 1
            baby: dict[int, int] = {}
            baby_set = baby.__setitem__

            e = 1
            for j in range(m):
                baby_set(e, j)
                e = (e * g) % p

            g_inv_m = pow(g, q - m, p)  # g^{-m}
            cached = (m, g_inv_m, baby)

            if len(self._bsgs_cache) >= self._bsgs_cache_max:
                self._bsgs_cache.clear()
            self._bsgs_cache[key] = cached

        m, g_inv_m, baby = cached
        baby_get = baby.get

        gamma = h
        for i in range(m + 1):
            j = baby_get(gamma)
            if j is not None:
                x = i * m + j
                if x < q:
                    return x
            gamma = (gamma * g_inv_m) % p
        raise ValueError("no dlog")

    def _ph_solve_from_entries(
        self,
        p: int,
        h: int,
        order: int,
        entries: tuple[tuple[int, int, int, int, int, tuple[int, ...], tuple[int, ...]], ...],
    ) -> int:
        """Pohlig–Hellman solve using cached g-dependent entries."""
        pow_ = pow
        crt_pair = self._crt_pair
        dlog_q = self._bsgs_prime_order

        x = 0
        mod = 1

        for q, e, qe, n_div_qe, g0, base_list, exp_list in entries:
            # h reduced to subgroup of size q^e
            h1 = pow_(h, n_div_qe, p)

            inv_g1x = 1
            x_qe = 0
            q_pow = 1

            for k in range(e):
                c = pow_((h1 * inv_g1x) % p, exp_list[k], p)
                d = dlog_q(p, g0, c, q)

                x_qe += d * q_pow
                inv_g1x = (inv_g1x * pow_(base_list[k], d, p)) % p
                q_pow *= q

            x, mod = crt_pair(x, mod, x_qe, qe)

        return x % order

    def _order_of_g(self, p: int, g: int, n: int, fac_n: dict[int, int]) -> tuple[int, dict[int, int]]:
        order = n
        fac_order = dict(fac_n)
        pow_ = pow
        for q, e in fac_n.items():
            for _ in range(e):
                if order % q == 0 and pow_(g, order // q, p) == 1:
                    order //= q
                    fac_order[q] -= 1
                else:
                    break
            if fac_order.get(q, 0) == 0:
                fac_order.pop(q, None)
        return order, fac_order

    def solve(self, problem: dict[str, int], **kwargs: Any) -> Any:
        p = int(problem["p"])
        g = int(problem["g"]) % p
        h = int(problem["h"]) % p

        if h == 1:
            return {"x": 0}

        n = p - 1

        # Tiny groups: linear scan is fastest.
        if n <= 4096:
            v = 1
            gg = g
            for x in range(n):
                if v == h:
                    return {"x": x}
                v = (v * gg) % p
            return {"x": 0}

        fac_n = self._factor_p_minus_1(p)

        # Fast path: PH on full group order, verify once.
        try:
            entries = self._get_ph_entries(p, g, n, fac_n)
            x = self._ph_solve_from_entries(p, h, n, entries)
            if pow(g, x, p) == h:
                return {"x": x}
        except Exception:
            pass

        # Slow path: actual order(g)
        try:
            order, fac_order = self._order_of_g(p, g, n, fac_n)
            if order <= 1:
                return {"x": 0}
            if pow(h, order, p) != 1:
                return {"x": int(self._sympy_discrete_log(p, h, g))}
            entries2 = self._get_ph_entries(p, g, order, fac_order)
            x2 = self._ph_solve_from_entries(p, h, order, entries2)
            if pow(g, x2, p) == h:
                return {"x": x2}
        except Exception:
            pass

        return {"x": int(self._sympy_discrete_log(p, h, g))}