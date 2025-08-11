import sympy
from sympy.ntheory.residue_ntheory import discrete_log

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the discrete logarithm x such that g^x ≡ h (mod p)
        using SymPy's optimized discrete_log implementation.
        Returns {"x": x}.
        """
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        # SymPy returns a Python integer (or sympy Integer); ensure plain int
        x = discrete_log(p, h, g)
        return {"x": int(x)}