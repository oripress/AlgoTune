import math
from typing import Any
import numpy as np
import cvxpy as cp
from scipy.special import xlogy
from numba import jit

# Cache log2 computation for efficiency
_LOG2 = math.log(2)
_LOG2_INV = 1.0 / _LOG2

@jit(nopython=True)
def _blahut_arimoto_step(P, x, n, m):
    """Single iteration of Blahut-Arimoto algorithm"""
    # Compute y = P @ x
    y = np.zeros(m)
    for i in range(m):
        for j in range(n):
            y[i] += P[i, j] * x[j]
    
    # Compute q_ij = P_ij * x_j / y_i
    q = np.zeros((m, n))
    for i in range(m):
        if y[i] > 1e-15:
            for j in range(n):
                q[i, j] = P[i, j] * x[j] / y[i]
    
    # Compute new x_j = product_i (q_ij ^ P_ij)
    x_new = np.ones(n)
    for j in range(n):
        for i in range(m):
            if P[i, j] > 0:
                x_new[j] *= q[i, j] ** P[i, j]
    
    # Normalize x
    x_sum = np.sum(x_new)
    if x_sum > 0:
        x_new = x_new / x_sum
    
    return x_new, y

@jit(nopython=True)
def _compute_mutual_information(P, x, y, n, m):
    """Compute mutual information given P, x, and y"""
    mi = 0.0
    log2_inv = 1.0 / math.log(2)
    
    for i in range(m):
        if y[i] > 1e-15:
            for j in range(n):
                if P[i, j] > 0 and x[j] > 1e-15:
                    mi += x[j] * P[i, j] * math.log(P[i, j] / y[i]) * log2_inv
    
    return mi

class Solver:
    def solve(self, problem: dict) -> dict:
        P = np.asarray(problem["P"], dtype=np.float64)
        m, n = P.shape
        
        # Input validation
        if not (n > 0 and m > 0):
            return None
        if not np.allclose(np.sum(P, axis=0), 1, atol=1e-6):
            return None
        
        # For very small problems, use Blahut-Arimoto with Numba
        if n <= 3 and m <= 3:
            return self._solve_blahut_arimoto(P, m, n)
        
        # For larger problems, use CVXPY
        return self._solve_cvxpy(P, m, n)
    
    def _solve_blahut_arimoto(self, P, m, n):
        """Use Blahut-Arimoto algorithm with Numba for very small problems"""
        # Initialize with uniform distribution
        x = np.ones(n, dtype=np.float64) / n
        
        # Blahut-Arimoto iterations
        for _ in range(50):
            x, y = _blahut_arimoto_step(P, x, n, m)
        
        # Compute final mutual information
        x, y = _blahut_arimoto_step(P, x, n, m)
        C = _compute_mutual_information(P, x, y, n, m)
        
        return {"x": x.tolist(), "C": C}
    
    def _solve_cvxpy(self, P, m, n):
        """Use CVXPY for larger problems"""
        # Optimized c vector computation
        P_safe = np.where(P > 0, P, 1e-15)
        c = np.sum(P_safe * np.log(P_safe), axis=0) * _LOG2_INV
        
        # Create optimization problem
        x = cp.Variable(shape=n, name="x")
        y = P @ x
        mutual_information = c @ x + cp.sum(cp.entr(y) * _LOG2_INV)
        objective = cp.Maximize(mutual_information)
        constraints = [cp.sum(x) == 1, x >= 0]
        
        prob = cp.Problem(objective, constraints)
        
        # Set warm start
        x.value = np.ones(n) / n
        
        # Try different solvers with more aggressive settings
        try:
            prob.solve(solver=cp.ECOS, verbose=False, max_iters=30, 
                      feastol=1e-7, abstol=1e-7, reltol=1e-7, warm_start=True)
        except Exception:
            try:
                prob.solve(solver=cp.OSQP, verbose=False, max_iter=60, 
                          eps_abs=1e-6, eps_rel=1e-6, warm_start=True)
            except Exception:
                try:
                    prob.solve(solver=cp.SCS, verbose=False, max_iters=60, 
                              eps=1e-6, warm_start=True)
                except Exception:
                    return None
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        if prob.value is None:
            return None
        
        return {"x": x.value.tolist(), "C": float(prob.value)}