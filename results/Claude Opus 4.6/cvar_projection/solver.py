import numpy as np
from scipy.optimize import minimize as sp_minimize

class Solver:
    def solve(self, problem, **kwargs):
        x0 = np.array(problem["x0"], dtype=np.float64)
        A = np.array(problem["loss_scenarios"], dtype=np.float64)
        beta = float(problem["beta"])
        kappa = float(problem["kappa"])
        
        m, n = A.shape
        k = int((1 - beta) * m)
        if k <= 0:
            return {"x_proj": x0.tolist()}
        
        alpha = kappa * k
        
        # Compute initial losses
        y0 = A @ x0
        
        # Check if already feasible
        if k >= m:
            top_k_sum = np.sum(y0)
        else:
            top_k_sum = np.sum(np.partition(y0, -k)[-k:])
        
        if top_k_sum <= alpha + 1e-10:
            return {"x_proj": x0.tolist()}
        
        # k >= m: simple halfspace projection
        if k >= m:
            c = np.sum(A, axis=0)
            cn2 = np.dot(c, c)
            if cn2 > 1e-20:
                x_proj = x0 - ((np.dot(c, x0) - alpha) / cn2) * c
                return {"x_proj": x_proj.tolist()}
            return {"x_proj": x0.tolist()}
        
        # Dual approach: x* = x0 - A^T lambda
        # where lambda >= 0, sum(lambda) free but lambda_i <= 1/k * mu, sum = mu
        # Actually: CVaR constraint sum_largest(Ax, k) <= alpha
        # Dual: minimize ||A^T lambda||^2/2 - x0^T A^T lambda
        #   s.t. sum(lambda) * alpha_effective, 0 <= lambda_i <= 1
        # 
        # The Lagrangian dual of projecting onto {x: sum_largest(Ax,k) <= alpha}:
        # x* = x0 - sum_i lambda_i * a_i where lambda in subdiff of sum_largest at Ax*
        # lambda_i in [0, 1], sum(lambda_i) = k, and lambda_i = 1 if y_i is in top-k,
        # lambda_i = 0 if not.
        #
        # Use SLSQP on the primal directly - it's the most reliable approach
        
        # Good warm start from active set
        idx = np.argpartition(y0, -k)[-k:]
        c = np.sum(A[idx], axis=0)
        cn2 = np.dot(c, c)
        if cn2 > 1e-20:
            lam = (np.dot(c, x0) - alpha) / cn2
            if lam > 0:
                start = x0 - lam * c
            else:
                start = x0.copy()
        else:
            start = x0.copy()
        
        def constraint_val(x):
            y = A @ x
            return alpha - np.sum(np.partition(y, -k)[-k:])
        
        def constraint_jac(x):
            y = A @ x
            idx2 = np.argpartition(y, -k)[-k:]
            return -np.sum(A[idx2], axis=0)
        
        result = sp_minimize(
            lambda x: np.sum((x - x0)**2),
            start,
            jac=lambda x: 2.0*(x - x0),
            method='SLSQP',
            constraints={'type': 'ineq', 'fun': constraint_val, 'jac': constraint_jac},
            options={'ftol': 1e-12, 'maxiter': 500}
        )
        
        return {"x_proj": result.x.tolist()}