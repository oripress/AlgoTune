import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the lp centering problem using CVXPY.

        :param problem: A dictionary of the lp centering problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution to the lp centering problem.

        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        c = np.array(problem["c"])
        A = np.array(problem["A"])
        b = np.array(problem["b"])
        n = c.shape[0]

        x = cp.Variable(n)
        prob = cp.Problem(cp.Maximize(c.T @ x - cp.sum(cp.log(x))), [A @ x == b])
        prob.solve(solver="HIGHS")
        assert prob.status == "optimal"
        return {"solution": x.value.tolist()}