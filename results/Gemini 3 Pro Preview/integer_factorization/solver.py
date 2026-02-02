import sympy
try:
    from factorizer import pollard_rho_brent
except ImportError:
    from sympy.ntheory import pollard_rho as pollard_rho_brent

class Solver:
    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        composite_val = problem["composite"]
        try:
            composite = int(composite_val)
        except (TypeError, ValueError):
            composite = int(str(composite_val))
            
        # Try our optimized pollard_rho_brent
        factor = pollard_rho_brent(composite)
        
        if factor is None or factor == composite or factor == 1:
            # Fallback to sympy if our implementation fails
            # This might happen if n is prime (though problem says composite)
            # or if the algorithm fails to find a factor
            factors_dict = sympy.factorint(composite)
            factors = [p for p, e in factors_dict.items() for _ in range(e)]
        else:
            p = factor
            q = composite // p
            factors = [p, q]
            
        factors.sort()
        return {"p": factors[0], "q": factors[1]}