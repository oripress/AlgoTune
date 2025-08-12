import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Reference CVXPY implementation of the optimal advertising problem.
        """
        # Extract problem parameters
        P = np.array(problem["P"])
        R = np.array(problem["R"])
        B = np.array(problem["B"])
        c = np.array(problem["c"])
        T = np.array(problem["T"])

        # Dimensions
        m, n = P.shape

        # Decision variable: displays
        D = cp.Variable((m, n))

        # Objective: maximize total revenue with caps
        revenue_per_ad = [
            cp.minimum(R[i] * P[i, :] @ D[i, :], B[i]) for i in range(m)
        ]
        total_revenue = cp.sum(revenue_per_ad)

        # Constraints
        constraints = [
            D >= 0,                        # non-negative displays
            cp.sum(D, axis=0) <= T,        # traffic capacity per slot
            cp.sum(D, axis=1) >= c,        # minimum displays per ad
        ]

        # Solve
        prob = cp.Problem(cp.Maximize(total_revenue), constraints)
        try:
            prob.solve()
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return {"status": prob.status, "optimal": False}

            # Extract solution
            D_val = D.value
            clicks = np.zeros(m)
            revenue = np.zeros(m)
            for i in range(m):
                clicks[i] = np.sum(P[i, :] * D_val[i, :])
                revenue[i] = min(R[i] * clicks[i], B[i])

            return {
                "status": prob.status,
                "optimal": True,
                "displays": D_val.tolist(),
                "clicks": clicks.tolist(),
                "revenue_per_ad": revenue.tolist(),
                "total_revenue": float(np.sum(revenue)),
                "objective_value": float(prob.value),
            }
        except cp.SolverError as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}