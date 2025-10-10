from sympy.ntheory.residue_ntheory import discrete_log
try:
    from dlog_cy import baby_step_giant_step
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

class Solver:
    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        """
        Solve the discrete logarithm problem.
        
        :param problem: A dictionary with keys "p", "g", and "h"
        :return: A dictionary with key "x" containing the discrete logarithm
        """
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        
        # For small p, use Cython-optimized baby-step giant-step
        if HAS_CYTHON and p < 10**12:
            x = baby_step_giant_step(g, h, p)
            if x != -1:
                return {"x": int(x)}
        
        # Use sympy's discrete_log for large p or as fallback
        return {"x": discrete_log(p, h, g)}