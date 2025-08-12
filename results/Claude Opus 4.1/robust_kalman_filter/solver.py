import numpy as np
import cvxpy as cp
from typing import Any

class Solver:
    def __init__(self):
        # Pre-compile solver settings for reuse
        self.solver_kwargs = {
            'solver': cp.CLARABEL,
            'tol_feas': 1e-6,
            'tol_gap_abs': 1e-6,
            'tol_gap_rel': 1e-6,
            'max_iter': 1000
        }
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Optimized robust Kalman filtering solver using Huber loss.
        """
        # Extract problem data with minimal overhead
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        C = np.asarray(problem["C"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        x0 = np.asarray(problem["x_initial"], dtype=np.float64)
        tau = problem["tau"]
        M = problem["M"]
        
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]
        
        # Create optimization variables
        x = cp.Variable((N + 1, n))
        w = cp.Variable((N, p))
        v = cp.Variable((N, m))
        
        # Build objective efficiently
        # Process noise term
        process_noise = cp.sum_squares(w)
        
        # Measurement noise with Huber penalty - vectorized computation
        huber_terms = []
        for t in range(N):
            huber_terms.append(cp.huber(cp.norm(v[t, :]), M))
        measurement_noise = tau * cp.sum(huber_terms)
        
        objective = cp.Minimize(process_noise + measurement_noise)
        
        # Build constraints efficiently
        constraints = []
        constraints.append(x[0] == x0)
        
        # Vectorized constraint construction
        for t in range(N):
            constraints.append(x[t + 1] == A @ x[t] + B @ w[t])
            constraints.append(y[t] == C @ x[t] + v[t])
        
        # Create and solve problem
        prob = cp.Problem(objective, constraints)
        
        try:
            # Use CLARABEL solver - often fastest for QP-like problems
            prob.solve(**self.solver_kwargs)
            
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                # Fallback to OSQP if CLARABEL fails
                prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, 
                          max_iter=2000, polish=True, adaptive_rho=True)
        except:
            try:
                # Final fallback to default solver
                prob.solve()
            except:
                return {"x_hat": [], "w_hat": [], "v_hat": []}
        
        # Check solution validity
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        
        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }