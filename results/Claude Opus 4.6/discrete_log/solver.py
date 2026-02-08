from math import isqrt
from sympy.ntheory import factorint
from sympy.ntheory.residue_ntheory import discrete_log as sympy_dlog
from sympy.ntheory.residue_ntheory import discrete_log as sympy_dlog

def bsgs(g, h, p, n):
    """Baby-step giant-step: find x in [0, n) such that g^x ≡ h (mod p)."""
    if n <= 0:
        return 0
    h = h % p
    if h == 1 % p:
        return 0

    m = isqrt(n)
    if m * m < n:
        m += 1

    table = {}
    pw = 1
    for j in range(m):
        table[pw] = j
        pw = pw * g % p

    inv_gm = pow(g, -m, p)

    gamma = h
    for i in range(m + 1):
        if gamma in table:
            val = i * m + table[gamma]
            return val % n
        gamma = gamma * inv_gm % p

    return None

def dlog_prime_power(g, h, p, q, e, pe):
    """Solve g^x ≡ h (mod p) where g has order q^e = pe."""
    gamma = pow(g, pe // q, p)
    g_inv = pow(g, -1, p)

    x = 0
    for k in range(e):
        exp = pe // (q ** (k + 1))
        gx_inv = pow(g_inv, x, p)
        h_k = pow(gx_inv * h % p, exp, p)

        if h_k == 1:
            d_k = 0
        else:
            d_k = bsgs(gamma, h_k, p, q)
            if d_k is None:
                return None

        x = x + d_k * (q ** k)

    return x % pe

class Solver:
    def __init__(self):
        factorint(100)

    def solve(self, problem, **kwargs):
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]

        h_mod = h % p
        if h_mod == 1 % p:
            return {"x": 0}
        if h_mod == g % p:
            return {"x": 1}

        p_minus_1 = p - 1
        if p_minus_1 <= 0:
            return {"x": 0}

        try:
            return self._pohlig_hellman(p, g, h, p_minus_1)
        except Exception:
            return {"x": int(sympy_dlog(p, h, g))}

    def _pohlig_hellman(self, p, g, h, p_minus_1):
        pm1_factors = factorint(p_minus_1)

        # Compute order of g
        order = p_minus_1
        for q in pm1_factors:
            while order % q == 0 and pow(g, order // q, p) == 1:
                order //= q

        if order <= 1:
            return {"x": 0}

        # Check h is in subgroup
        if pow(h, order, p) != 1:
            raise ValueError("h not in subgroup")

        # Factor the order
        order_factors = {}
        remaining = order
        for q in pm1_factors:
            if remaining % q == 0:
                e = 0
                while remaining % q == 0:
                    remaining //= q
                    e += 1
                order_factors[q] = e
        if remaining > 1:
            order_factors[remaining] = 1

        crt_a = []
        crt_m = []

        for q, e in order_factors.items():
            q = int(q)
            e = int(e)
            pe = q ** e
            gi = pow(g, order // pe, p)
            hi = pow(h, order // pe, p)

            if hi == 1:
                crt_a.append(0)
            elif e == 1:
                r = bsgs(gi, hi, p, q)
                if r is None:
                    raise ValueError("bsgs failed")
                crt_a.append(r)
            else:
                r = dlog_prime_power(gi, hi, p, q, e, pe)
                if r is None:
                    raise ValueError("dlog_prime_power failed")
                crt_a.append(r)
            crt_m.append(pe)

        if len(crt_a) == 0:
            return {"x": 0}

        x = crt_a[0] % crt_m[0]
        mod = crt_m[0]
        for i in range(1, len(crt_a)):
            r = crt_a[i] % crt_m[i]
            mi = crt_m[i]
            diff = (r - x) % mi
            t = diff * pow(mod, -1, mi) % mi
            x = x + mod * t
            mod *= mi

        x = int(x % order)
        if pow(g, x, p) != h % p:
            raise ValueError("verification failed")
        return {"x": x}