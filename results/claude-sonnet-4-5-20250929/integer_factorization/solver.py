import sympy
from pollard import pollard_rho_cython

class Solver:
    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        """
        Factor using optimized Pollard's rho with fallback to sympy.
        """
        n = int(problem["composite"])
        
        # Quick trial division for small factors
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            if n % p == 0:
                q = n // p
                return {"p": p, "q": q}
        
        # Try Cython Pollard's rho for medium-sized factors
        try:
            factor = pollard_rho_cython(n)
            if factor and 1 < factor < n:
                p, q = factor, n // factor
                if p > q:
                    p, q = q, p
                return {"p": int(p), "q": int(q)}
        except:
            pass
        
        # Fallback to sympy for hard cases
        factors = sympy.factorint(n)
        factor_list = []
        for prime, exp in factors.items():
            for _ in range(exp):
                factor_list.append(int(prime))
        
        if len(factor_list) != 2:
            raise ValueError(f"Expected 2 factors, got {len(factor_list)}")
        
        p, q = sorted(factor_list)
        return {"p": p, "q": q}