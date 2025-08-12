import math
import random
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """Your implementation goes here."""
        # Get the composite number
        if isinstance(problem, dict):
            composite_val = int(problem["composite"])
        elif isinstance(problem, int):
            composite_val = problem
        else:
            composite_val = int(problem)
        
        # Handle small factors first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if composite_val % p == 0:
                q = composite_val // p
                if self._is_prime(q):
                    return {"p": p, "q": q}
        
        # For larger numbers, use Pollard's Rho algorithm
        if composite_val < 1000000:
            # Use trial division for smaller numbers
            limit = math.isqrt(composite_val) + 1
            f = 53  # Start from where we left off
            while f <= limit:
                if composite_val % f == 0:
                    q = composite_val // f
                    if self._is_prime(q):
                        return {"p": f, "q": q}
                f += 2
        else:
            # Use Pollard's Rho for larger numbers
            p = self._pollards_rho(composite_val)
            if p is not None:
                q = composite_val // p
                if self._is_prime(q):
                    return {"p": min(p, q), "q": max(p, q)}
        
        # If we get here, the number itself is prime (shouldn't happen for our problems)
        return {"p": 1, "q": composite_val}
    
    def _pollards_rho(self, n):
        """Pollard's Rho algorithm for factorization."""
        if n % 2 == 0:
            return 2
        
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
            
            if d == n:
                return self._pollards_rho(n)
        
        return d
    
    def _is_prime(self, n):
        """Check if a number is prime using trial division."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        limit = math.isqrt(n) + 1
        f = 3
        while f <= limit:
            if n % f == 0:
                return False
            f += 2
        return True