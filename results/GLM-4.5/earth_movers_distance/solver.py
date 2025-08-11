import numpy as np
import ot

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """
        Solve the EMD problem using ot.lp.emd with minimal overhead.

        :param problem: A dictionary representing the EMD problem.
        :return: A dictionary with key "transport_plan" containing the optimal
                 transport plan matrix G as a list of lists.

        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        a = problem["source_weights"]
        b = problem["target_weights"]
        M = problem["cost_matrix"]

        # Minimal data preparation
        M_cont = np.ascontiguousarray(M, dtype=np.float64)

        # Compute the optimal transport plan
        G = ot.lp.emd(a, b, M_cont, check_marginals=False)

        solution = {"transport_plan": G}
        return solution