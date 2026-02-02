import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs):
        x0 = np.array(problem["x0"], dtype=np.float64)
        A = np.array(problem["loss_scenarios"], dtype=np.float64)
        beta = float(problem.get("beta", 0.95))
        kappa = float(problem.get("kappa", 1.0))
        
        n_scenarios, n_dims = A.shape
        k = int((1 - beta) * n_scenarios)
        
        if k == 0:
            return {"x_proj": x0.tolist()}
        
        alpha = kappa * k
        
        # Check if x0 already satisfies the constraint
        losses = A @ x0
        if n_scenarios > k:
            top_k = np.partition(losses, n_scenarios - k)[n_scenarios - k:]
            top_k_sum = np.sum(top_k)
        else:
            top_k_sum = np.sum(losses)
        
        if top_k_sum <= alpha + 1e-8:
            return {"x_proj": x0.tolist()}
        
        # Solve the projection problem with reformulation
        x = cp.Variable(n_dims)
        t = cp.Variable()
        u = cp.Variable(n_scenarios, nonneg=True)
        
        objective = cp.Minimize(cp.sum_squares(x - x0))
        constraints = [
            k * t + cp.sum(u) <= alpha,
            u >= A @ x - t,
        ]
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-6, tol_gap_rel=1e-6)
            if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and x.value is not None:
                return {"x_proj": x.value.tolist()}
            
            prob.solve(solver=cp.ECOS)
            if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and x.value is not None:
                return {"x_proj": x.value.tolist()}
            
            return {"x_proj": []}
        except Exception:
            try:
                prob.solve(solver=cp.ECOS)
                if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and x.value is not None:
                    return {"x_proj": x.value.tolist()}
            except:
                pass
            return {"x_proj": []}