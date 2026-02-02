import numpy as np
import osqp
from scipy import sparse

class Solver:
    def solve(self, problem, **kwargs):
        mu = np.asarray(problem["μ"], dtype=float)
        Sigma = np.asarray(problem["Σ"], dtype=float)
        gamma = float(problem["γ"])
        n = mu.size
        
        if n == 1:
            return {"w": [1.0]}
        
        P = sparse.csc_matrix(2 * gamma * Sigma)
        q = -mu
        
        A_eq = sparse.csc_matrix(np.ones((1, n)))
        A_ineq = sparse.eye(n, format='csc')
        A_full = sparse.vstack([A_eq, A_ineq], format='csc')
        l = np.hstack([[1.0], np.zeros(n)])
        u = np.hstack([[1.0], np.full(n, np.inf)])
        
        solver = osqp.OSQP()
        solver.setup(P=P, q=q, A=A_full, l=l, u=u, 
                    verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                    polish=False, max_iter=1000, check_termination=25,
                    scaling=10, rho=0.1)
        
        result = solver.solve()
        
        if result.x is None or not np.isfinite(result.x).all():
            return {"w": (np.ones(n) / n).tolist()}
        
        w = np.maximum(result.x, 0)
        s = np.sum(w)
        if s > 1e-10:
            w = w / s
        return {"w": w.tolist()}