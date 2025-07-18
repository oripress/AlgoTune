import math
import numpy as np
import cvxpy as cp
from scipy.special import xlogy

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        # Parse input
        P_list = problem.get("P")
        if P_list is None:
            return None
        P = np.array(P_list, dtype=np.float64)
        if P.ndim != 2:
            return None
        m, n = P.shape
        if m < 1 or n < 1 or not np.allclose(P.sum(axis=0), 1.0, atol=1e-6):
            return None

        # Define variables
        x = cp.Variable(n, nonneg=True)
        y = P @ x

        # Precompute c = sum_i P_ij log2(P_ij)
        c = np.sum(xlogy(P, P), axis=0) / math.log(2)

        # Mutual information objective: c^T x + sum(entr(y))/log(2)
        mutual_information = c @ x + cp.sum(cp.entr(y)) / math.log(2)

        # Setup and solve problem
        prob = cp.Problem(cp.Maximize(mutual_information),
                          [cp.sum(x) == 1])
        try:
            prob.solve()
        except cp.SolverError:
            return None
        except Exception:
            return None

        # Check solution status
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        if prob.value is None:
            return None

        # Return optimal distribution and capacity
        return {"x": x.value.tolist(), "C": prob.value}