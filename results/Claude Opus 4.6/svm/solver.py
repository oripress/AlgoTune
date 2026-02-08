import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem, **kwargs):
        X = np.array(problem["X"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        C = float(problem["C"])
        n, p = X.shape
        
        try:
            import osqp
            return self._solve_osqp(X, y, C, n, p)
        except Exception:
            return self._solve_cvxpy(X, y, C, n, p)
    
    def _solve_osqp(self, X, y, C, n, p):
        import osqp
        dim = p + 1 + n
        
        # Build P matrix (sparse diagonal)
        P = sparse.diags(
            np.concatenate([np.ones(p), np.zeros(1 + n)]),
            format='csc'
        )
        
        # Build q vector
        q = np.concatenate([np.zeros(p + 1), C * np.ones(n)])
        
        # Build constraint matrix A
        yX = y[:, None] * X
        
        A = sparse.vstack([
            sparse.hstack([
                sparse.csc_matrix(yX),
                sparse.csc_matrix(y.reshape(-1, 1)),
                sparse.eye(n, format='csc')
            ], format='csc'),
            sparse.hstack([
                sparse.csc_matrix((n, p + 1)),
                sparse.eye(n, format='csc')
            ], format='csc')
        ], format='csc')
        
        l = np.concatenate([np.ones(n), np.zeros(n)])
        u = np.full(2 * n, np.inf)
        
        solver = osqp.OSQP()
        solver.setup(
            P, q, A, l, u,
            verbose=False,
            eps_abs=1e-7,
            eps_rel=1e-7,
            max_iter=10000,
            polish=True
        )
        result = solver.solve()
        
        if result.info.status not in ['solved', 'solved_inaccurate']:
            raise RuntimeError("OSQP failed to solve")
        
        beta = result.x[:p]
        beta0 = result.x[p]
        
        # Compute optimal value from primal variables
        margins = y * (X @ beta + beta0)
        xi = np.maximum(0.0, 1.0 - margins)
        optimal_value = 0.5 * np.dot(beta, beta) + C * np.sum(xi)
        
        pred = X @ beta + beta0
        missclass = np.mean((pred * y) < 0)
        
        return {
            "beta0": float(beta0),
            "beta": beta.tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass),
        }
    
    def _solve_cvxpy(self, X, y, C, n, p):
        import cvxpy as cp
        
        y_col = y[:, None]
        beta_var = cp.Variable((p, 1))
        beta0_var = cp.Variable()
        xi_var = cp.Variable((n, 1))
        
        objective = cp.Minimize(
            0.5 * cp.sum_squares(beta_var) + C * cp.sum(xi_var)
        )
        constraints = [
            xi_var >= 0,
            cp.multiply(y_col, X @ beta_var + beta0_var) >= 1 - xi_var,
        ]
        
        prob = cp.Problem(objective, constraints)
        optimal_value = prob.solve()
        
        if beta_var.value is None:
            return None
        
        beta_val = beta_var.value.flatten()
        beta0_val = float(beta0_var.value)
        
        pred = X @ beta_val + beta0_val
        missclass = np.mean((pred * y) < 0)
        
        return {
            "beta0": beta0_val,
            "beta": beta_val.tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass),
        }