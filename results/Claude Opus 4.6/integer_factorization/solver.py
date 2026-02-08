import math
import random
from typing import Any


class Solver:
    def __init__(self):
        # Precompute small primes for trial division and p-1 method
        self.small_primes = self._sieve(1000000)
    
    def _sieve(self, limit):
        """Sieve of Eratosthenes."""
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    def solve(self, problem, **kwargs) -> Any:
        n = problem["composite"]
        
        if n <= 1:
            raise ValueError("Cannot factor")
        
        # Check if it's a perfect square
        s = math.isqrt(n)
        if s * s == n:
            return {"p": s, "q": s}
        
        # Try small primes first via trial division
        factor = self._trial_division(n)
        if factor is not None:
            p, q = sorted([factor, n // factor])
            return {"p": p, "q": q}
        
        # Determine bit size to choose strategy
        bit_len = n.bit_length()
        
        # Try Pollard's p-1 method (fast when p-1 has small factors)
        factor = self._pollard_pm1(n)
        if factor is not None:
            p, q = sorted([factor, n // factor])
            return {"p": p, "q": q}
        
        # For small to medium numbers, Pollard's rho is fast enough
        if bit_len <= 128:
            factor = self._pollard_rho_brent(n)
            if factor is not None:
                p, q = sorted([factor, n // factor])
                return {"p": p, "q": q}
        else:
            # Try rho first with limited attempts, then ECM
            factor = self._pollard_rho_brent(n, max_attempts=20)
            if factor is not None:
                p, q = sorted([factor, n // factor])
                return {"p": p, "q": q}
            
            # ECM for larger numbers
            factor = self._ecm(n)
            if factor is not None:
                p, q = sorted([factor, n // factor])
                return {"p": p, "q": q}
        
        # Fallback to sympy
        import sympy
        factors = [prime for prime, exp in sympy.factorint(n).items() for _ in range(exp)]
        p, q = sorted(map(int, factors))
        return {"p": p, "q": q}
    
    def _trial_division(self, n):
        for p in self.small_primes:
            if p * p > n:
                return None
            if n % p == 0:
                return p
        return None
    
    def _pollard_pm1(self, n, B1=100000):
        """Pollard's p-1 method."""
        a = 2
        for p in self.small_primes:
            if p > B1:
                break
            pk = p
            while pk * p <= B1:
                pk *= p
            a = pow(a, pk, n)
        
        g = math.gcd(a - 1, n)
        if 1 < g < n:
            return g
        return None
    
    def _pollard_rho_brent(self, n, max_attempts=200):
        """Pollard's rho with Brent's cycle detection and batch GCD."""
        if n <= 1:
            return None
        if n % 2 == 0:
            return 2
        
        for attempt in range(max_attempts):
            c = random.randrange(1, n)
            y = random.randrange(1, n)
            r = 1
            q = 1
            g = 1
            
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                
                k = 0
                while k < r and g == 1:
                    ys = y
                    batch = min(128, r - k)
                    for _ in range(batch):
                        y = (y * y + c) % n
                        q = q * abs(x - y) % n
                    g = math.gcd(q, n)
                    k += batch
                r *= 2
                
                if r > 1 << 26:
                    break
            
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                
            if 1 < g < n:
                return g
        
        return None
    
    def _ecm_add(self, p1, p2, p0, n):
        """Add two points on Montgomery curve (projective coords)."""
        # p1 = (x1, z1), p2 = (x2, z2), p0 = (x0, z0) = p1 - p2
        x1, z1 = p1
        x2, z2 = p2
        x0, z0 = p0
        u = (x1 - z1) * (x2 + z2)
        v = (x1 + z1) * (x2 - z2)
        add_ = u + v
        sub_ = u - v
        x = z0 * (add_ * add_) % n
        z = x0 * (sub_ * sub_) % n
        return (x, z)
    
    def _ecm_double(self, p, n, a24):
        """Double a point on Montgomery curve."""
        x, z = p
        u = (x + z) * (x + z)
        v = (x - z) * (x - z)
        diff = u - v
        x2 = (u * v) % n
        z2 = (diff * (v + a24 * diff)) % n
        return (x2, z2)
    
    def _ecm_multiply(self, k, p, n, a24):
        """Scalar multiplication using Montgomery ladder."""
        if k == 0:
            return (0, 0)
        if k == 1:
            return p
        
        # Binary method (Montgomery ladder)
        r0 = p
        r1 = self._ecm_double(p, n, a24)
        
        bits = bin(k)[3:]  # Skip '0b1'
        for bit in bits:
            if bit == '1':
                r0 = self._ecm_add(r0, r1, p, n)
                r1 = self._ecm_double(r1, n, a24)
            else:
                r1 = self._ecm_add(r0, r1, p, n)
                r0 = self._ecm_double(r0, n, a24)
        
        return r0
    
    def _ecm(self, n, curves=100, B1=10000, B2=1000000):
        """Lenstra's Elliptic Curve Method."""
        for _ in range(curves):
            # Random curve and point
            sigma = random.randrange(6, n)
            u = (sigma * sigma - 5) % n
            v = (4 * sigma) % n
            
            x0 = pow(u, 3, n)
            z0 = pow(v, 3, n)
            
            a_num = (pow(v - u, 3, n) * (3 * u + v)) % n
            a_den = (16 * x0 * v) % n
            
            g = math.gcd(a_den, n)
            if 1 < g < n:
                return g
            if g == n:
                continue
            
            a_den_inv = pow(a_den, -1, n)
            a24 = (a_num * a_den_inv - 2) % n
            # a24 = (a + 2) / 4
            g = math.gcd(4, n)
            if g > 1 and g < n:
                return g
            a24 = ((a_num * a_den_inv - 2) * pow(4, -1, n)) % n
            
            # Wait, let me redo this properly
            # For Montgomery curve By^2 = x^3 + Ax^2 + x
            # a24 = (A + 2) / 4
            # A = (a_num * a_den_inv) where we computed it
            A = (a_num * a_den_inv) % n
            a24 = ((A + 2) * pow(4, -1, n)) % n
            
            point = (x0 % n, z0 % n)
            
            # Stage 1
            try:
                for p in self.small_primes:
                    if p > B1:
                        break
                    pk = p
                    while pk <= B1:
                        point = self._ecm_multiply(p, point, n, a24)
                        pk *= p
            except (ValueError, ZeroDivisionError):
                continue
            
            g = math.gcd(point[1], n)
            if 1 < g < n:
                return g
            if g == n:
                continue
            
            # Stage 2 (simplified)
            # Use baby-step giant-step approach
            try:
                Q = point
                prev_x, prev_z = Q
                for p in self.small_primes:
                    if p <= B1:
                        continue
                    if p > B2:
                        break
                    Q = self._ecm_multiply(p, point, n, a24)
                    g = math.gcd(Q[1], n)
                    if 1 < g < n:
                        return g
                    if g == n:
                        break
            except (ValueError, ZeroDivisionError):
                continue
        
        return None