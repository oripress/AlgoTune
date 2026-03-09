import math
from typing import Any

try:
    import fastfactor
except Exception:
    fastfactor = None

class Solver:
    @staticmethod
    def _rho_brent(n: int) -> int:
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5

        gcd = math.gcd
        seed = (n & 127) + 1

        for attempt in range(1, 10):
            y = attempt + 1
            c = seed + (attempt << 1) - 1
            if c >= n:
                c %= n
            if c == 0:
                c = 1

            m = 128
            g = 1
            r = 1

            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n

                q = 1
                k = 0
                while k < r and g == 1:
                    ys = y
                    limit = m if m < (r - k) else (r - k)
                    for _ in range(limit):
                        y = (y * y + c) % n
                        diff = x - y
                        if diff < 0:
                            diff = -diff
                        q = (q * diff) % n
                    g = gcd(q, n)
                    k += limit
                r <<= 1

            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    diff = x - ys
                    if diff < 0:
                        diff = -diff
                    g = gcd(diff, n)
                    if g > 1:
                        break

            if 1 < g < n:
                return g

        for c in (1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43):
            x = 2
            y = 2
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                diff = x - y
                if diff < 0:
                    diff = -diff
                d = gcd(diff, n)
            if d != n:
                return d

        return n

    def solve(self, problem, **kwargs) -> Any:
        n = int(problem["composite"])

        if fastfactor is not None:
            p, q = fastfactor.factor_semiprime(n)
            if p != 1:
                return {"p": int(p), "q": int(q)}

        d = self._rho_brent(n)
        if d == n:
            import sympy

            factors = [int(prime) for prime, exp in sympy.factorint(n).items() for _ in range(exp)]
            p, q = sorted(factors)
            return {"p": p, "q": q}

        q = n // d
        if d > q:
            d, q = q, d
        return {"p": d, "q": q}