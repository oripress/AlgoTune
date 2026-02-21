from typing import Any
import numpy as np
from scipy.optimize import minimize
from numba import njit
@njit(fastmath=True, cache=True)
def objective_and_grad(w_flat, A, B, C, y, x0, tau, M, N, n, p, m, grad_w_flat, x, v, v_norms, lambda_adj):
    w = w_flat.reshape((N, p))
    grad_w = grad_w_flat.reshape((N, p))
    
    x[0] = x0
    
    obj = 0.0
    
    for t in range(N):
        v_t = y[t] - C @ x[t]
        v[t] = v_t
        
        v_norm_sq = 0.0
        for i in range(m):
            v_norm_sq += v_t[i]**2
        v_norm = np.sqrt(v_norm_sq)
        v_norms[t] = v_norm
        
        if v_norm <= M:
            obj += tau * v_norm_sq
        else:
            obj += tau * (2 * M * v_norm - M**2)
            
        for i in range(p):
            obj += w[t, i]**2
            
        x[t+1] = A @ x[t] + B @ w[t]
        
    for i in range(n):
        lambda_adj[i] = 0.0
    
    for t in range(N - 1, -1, -1):
        v_norm = v_norms[t]
        v_t = v[t]
        
        if v_norm <= M:
            grad_v = 2 * tau * v_t
        else:
            if v_norm > 0:
                grad_v = 2 * tau * M * v_t / v_norm
            else:
                grad_v = np.zeros(m)
                
        grad_w[t] = 2 * w[t] + B.T @ lambda_adj
        
        grad_x_meas = -C.T @ grad_v
        lambda_adj = A.T @ lambda_adj + grad_x_meas
        
    return obj, grad_w_flat
class Solver:
    def __init__(self):
        A = np.eye(1, dtype=np.float64)
        B = np.eye(1, dtype=np.float64)
        C = np.eye(1, dtype=np.float64)
        y = np.zeros((1, 1), dtype=np.float64)
        x0 = np.zeros(1, dtype=np.float64)
        w_flat = np.zeros(1, dtype=np.float64)
        grad_w_flat = np.zeros(1, dtype=np.float64)
        x = np.zeros((2, 1), dtype=np.float64)
        v = np.zeros((1, 1), dtype=np.float64)
        v_norms = np.zeros(1, dtype=np.float64)
        lambda_adj = np.zeros(1, dtype=np.float64)
        objective_and_grad(w_flat, A, B, C, y, x0, 1.0, 1.0, 1, 1, 1, 1, grad_w_flat, x, v, v_norms, lambda_adj)

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        C = np.array(problem["C"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        x0 = np.array(problem["x_initial"], dtype=np.float64)
        tau = float(problem["tau"])
        M = float(problem["M"])

        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        w0 = np.zeros(N * p, dtype=np.float64)
        grad_w_flat = np.zeros(N * p, dtype=np.float64)
        x = np.zeros((N + 1, n), dtype=np.float64)
        v = np.zeros((N, m), dtype=np.float64)
        v_norms = np.zeros(N, dtype=np.float64)
        lambda_adj = np.zeros(n, dtype=np.float64)
        
        res = minimize(
            objective_and_grad, 
            w0, 
            args=(A, B, C, y, x0, tau, M, N, n, p, m, grad_w_flat, x, v, v_norms, lambda_adj),
            method='L-BFGS-B', 
            jac=True,
            options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 1000}
        )
        
        w_opt = res.x.reshape((N, p))
        
        x_opt = np.zeros((N + 1, n))
        x_opt[0] = x0
        for t in range(N):
            x_opt[t+1] = A @ x_opt[t] + B @ w_opt[t]
            
        v_opt = np.zeros((N, m))
        for t in range(N):
            v_opt[t] = y[t] - C @ x_opt[t]
            
        return {
            "x_hat": x_opt.tolist(),
            "w_hat": w_opt.tolist(),
            "v_hat": v_opt.tolist()
        }