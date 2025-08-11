from sympy.ntheory.residue_ntheory import discrete_log

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the discrete logarithm problem.
        
        :param problem: A dictionary with keys "p", "g", "h"
        :return: A dictionary with key "x" containing the discrete logarithm
        """
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        
        # Use sympy's discrete_log directly
        return {"x": discrete_log(p, h, g)}