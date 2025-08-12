import random
import math

class Solver:
    def solve(self, problem, **kwargs):
        """Solve integer factorization using trial division and Pollard's rho."""
        composite = int(problem["composite"])
        composite = int(problem["composite"])

        # Handle small factors first
        if composite % 2 == 0:
            return {"p": 2, "q": composite // 2}

        # Trial division for small primes
        small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            if composite % p == 0:
                return {"p": p, "q": composite // p}

        # Pollard's rho algorithm with improvements
        def pollard_rho(n):
            if n % 2 == 0:
                return 2

            x = 2
            y = 2
            c = 1
            f = lambda x: (x * x + c) % n
            d = 1

            # Limit iterations to prevent infinite loops
            for _ in range(min(10000, n // 1000 + 100)):
                x = f(x)
                y = f(f(y))
                d = math.gcd(abs(x - y), n)

                if d > 1 and d < n:
                    return d

            return None  # Failed to find factor

        # Simple primality test
        def is_prime(n):
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            try:
                limit = int(n**0.5) + 1
            except OverflowError:
                limit = 1000
            for i in range(3, min(limit, 1000), 2):
                if n % i == 0:
                    return False
            return True

        # Trial division helper
        def trial_division(n):
            try:
                limit = min(10000, int(n**0.5) + 1)
            except OverflowError:
                limit = 10000
            for i in range(3, limit, 2):
                if n % i == 0:
                    return i
            return None

        # First try trial division
        factor = trial_division(composite)
        if factor is not None:
            p, q = factor, composite // factor
            # Validate that both are prime
            if is_prime(p) and is_prime(q):
                if p > q:
                    p, q = q, p
                return {"p": p, "q": q}

        # Then try Pollard's rho
        factor = pollard_rho(composite)
        if factor is not None:
            p, q = factor, composite // factor
            # Validate that both are prime
            if is_prime(p) and is_prime(q):
                if p > q:
                    p, q = q, p
                return {"p": p, "q": q}

        # For small numbers, use brute force as last resort
        # For small numbers, use brute force as last resort
        if composite < 1000000:
            try:
                limit = int(composite**0.5) + 1
            except OverflowError:
                limit = 1000
            for i in range(3, limit, 2):
                if composite % i == 0:
                    p, q = i, composite // i
                    if is_prime(p) and is_prime(q):
                        if p > q:
                            p, q = q, p
                        return {"p": p, "q": q}

        raise ValueError("Could not factor the composite number")