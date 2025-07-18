import math

class Solver:
    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        """
        Solve the integer factorization problem using Fermat's factorization method.
        This is efficient when the two prime factors are close to each other.
        """
        n = int(problem["composite"])
        
        # For two primes p and q that are close, they are both near sqrt(n)
        # Fermat's method: n = a² - b² = (a+b)(a-b) where a = (p+q)/2, b = (q-p)/2
        
        # Start with a = ceil(sqrt(n))
        a = int(math.isqrt(n))
        if a * a < n:
            a += 1
        
        # Find b such that a² - n = b²
        while True:
            b_squared = a * a - n
            b = int(math.isqrt(b_squared))
            
            if b * b == b_squared:
                # Found factorization: n = (a+b)(a-b)
                p = a - b
                q = a + b
                return {"p": p, "q": q}
            
            a += 1