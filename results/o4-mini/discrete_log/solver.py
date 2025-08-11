from sympy.ntheory.residue_ntheory import discrete_log

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Solve the discrete logarithm problem using Sympy's optimized algorithms.
        """
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        # discrete_log(p, h, g) finds x such that g^x ≡ h (mod p)
        x = discrete_log(p, h, g)
        return {"x": int(x)}