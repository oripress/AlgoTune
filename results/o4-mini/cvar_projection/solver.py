import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs):
        # Extract and validate inputs
        x0 = np.asarray(problem.get("x0", []), dtype=float)
        A = np.asarray(problem.get("loss_scenarios", []), dtype=float)
        if x0.size == 0 or A.size == 0:
            return {"x_proj": []}
        if A.ndim != 2 or x0.ndim != 1 or x0.shape[0] != A.shape[1]:
            return {"x_proj": []}
        n_scenarios, n_dims = A.shape

        # CVaR parameters
        beta = float(problem.get("beta", 0.95))
        kappa = float(problem.get("kappa", 0.0))
        k = int((1.0 - beta) * n_scenarios)
        if k < 1:
            return {"x_proj": x0.tolist()}
        alpha = kappa * k

        # Quick feasibility check
        losses0 = A.dot(x0)
        if np.partition(losses0, -k)[-k:].sum() <= alpha + 1e-8:
            return {"x_proj": x0.tolist()}

        # Setup and solve QP with CVXPY ECOS
        x = cp.Variable(n_dims)
        obj = cp.Minimize(cp.sum_squares(x - x0))
        constraints = [cp.sum_largest(A @ x, k) <= alpha]
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(
                solver=cp.ECOS,
                warm_start=True,
                abstol=1e-4,
                reltol=1e-4,
                feastol=1e-4,
                max_iters=10000,
                verbose=False
            )
        except Exception:
            try:
                prob.solve(
                    solver=cp.SCS,
                    warm_start=True,
                    eps=1e-4,
                    max_iters=10000,
                    verbose=False
                )
            except Exception:
                return {"x_proj": []}

        x_val = x.value
        if x_val is None or np.any(np.isnan(x_val)):
            return {"x_proj": []}
        return {"x_proj": x_val.tolist()}