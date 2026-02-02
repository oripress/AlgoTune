import math
import random
from typing import Any

class Solver:
    def __init__(self):
        # Precompute small primes for trial division
        limit = 100000
        sieve = [True] * limit
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit, i):
                    sieve[j] = False
        self.small_primes = [i for i in range(limit) if sieve[i]]
    
    def solve(self, problem: dict, **kwargs) -> Any:
        n = int(problem["composite"])
        
        # Trial division for small factors
        for p in self.small_primes:
            if p * p > n:
                break
            if n % p == 0:
                q = n // p
                return {"p": p, "q": q} if p < q else {"p": q, "q": p}
        
        # Use Pollard's rho with Brent's improvement
        factor = self._pollard_rho_brent(n)
        p, q = factor, n // factor
        return {"p": int(p), "q": int(q)} if p < q else {"p": int(q), "q": int(p)}
    
    def _pollard_rho_brent(self, n):
        if n % 2 == 0:
            return 2
        
        y = random.randint(1, n - 1)
        c = random.randint(1, n - 1)
        m = random.randint(1, n - 1)
        g, r, q = 1, 1, 1
        
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        
        return g if g != n else self._pollard_rho_brent(n)