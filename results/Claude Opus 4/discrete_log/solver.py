from sympy.ntheory.residue_ntheory import discrete_log

class Solver:
    def solve(self, problem):
        """Solve the discrete logarithm problem: find x such that g^x ≡ h (mod p)"""
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        
        # For very small p, brute force is actually faster
        if p <= 100:
            for x in range(p):
                if pow(g, x, p) == h:
                    return {"x": x}
        
        # For larger p, use sympy's optimized implementation
        return {"x": discrete_log(p, h, g)}