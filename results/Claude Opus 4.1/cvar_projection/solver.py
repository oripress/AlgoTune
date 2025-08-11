import cvxpy as cp
import numpy as np

class Solver:
    def __init__(self):
        self.beta = 0.95  # default beta
        self.kappa = 1.0  # default kappa
    
    def solve(self, problem: dict) -> dict:
        """
        Compute the projection onto the CVaR constraint set.
        """
        # Extract problem data
        x0 = np.array(problem["x0"])
        A = np.array(problem["loss_scenarios"])
        beta = float(problem.get("beta", self.beta))
        kappa = float(problem.get("kappa", self.kappa))

        n_scenarios, n_dims = A.shape

        # Define variables
        x = cp.Variable(n_dims)

        # Define objective: minimize distance to x0
        objective = cp.Minimize(cp.sum_squares(x - x0))

        # Add CVaR constraint
        k = int((1 - beta) * n_scenarios)
        alpha = kappa * k
        constraints = [cp.sum_largest(A @ x, k) <= alpha]

        # Define and solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
                return {"x_proj": []}

            return {"x_proj": x.value.tolist()}

        except cp.SolverError as e:
            return {"x_proj": []}
        except Exception as e:
            return {"x_proj": []}