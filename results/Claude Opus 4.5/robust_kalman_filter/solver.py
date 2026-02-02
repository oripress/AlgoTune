import numpy as np
from scipy.optimize import minimize
from numba import njit

@njit(cache=True)
def compute_obj_grad_numba(w_flat, A, B, C, y, x0, tau, M, N, n, p, m):
    w = w_flat.reshape((N, p))
    
    # Forward pass: compute x and v
    x = np.zeros((N + 1, n))
    v = np.zeros((N, m))
    x[0] = x0.copy()
    
    obj = 0.0
    for t in range(N):
        x[t + 1] = A @ x[t] + B @ w[t]
        v[t] = y[t] - C @ x[t]
        obj += np.sum(w[t] ** 2)
        v_norm = np.sqrt(np.sum(v[t] ** 2))
        if v_norm <= M:
            obj += tau * v_norm ** 2
        else:
            obj += tau * (2.0 * M * v_norm - M ** 2)
    
    # Compute dL_dv for each t
    dL_dv = np.zeros((N, m))
    for t in range(N):
        v_norm = np.sqrt(np.sum(v[t] ** 2))
        if v_norm < 1e-10:
            pass
        elif v_norm <= M:
            dL_dv[t] = 2.0 * tau * v[t]
        else:
            dL_dv[t] = 2.0 * tau * M * v[t] / v_norm
    
    # Backward pass (adjoint method)
    lambd = np.zeros(n)
    grad = np.zeros((N, p))
    
    for t in range(N - 1, -1, -1):
        grad[t] = 2.0 * w[t] + B.T @ lambd
        lambd = -C.T @ dL_dv[t] + A.T @ lambd
    
    return obj, grad.flatten()

@njit(cache=True)
def reconstruct_solution(w_flat, A, B, C, y, x0, N, n, p, m):
    w = w_flat.reshape((N, p))
    x = np.zeros((N + 1, n))
    v = np.zeros((N, m))
    x[0] = x0.copy()
    for t in range(N):
        x[t + 1] = A @ x[t] + B @ w[t]
        v[t] = y[t] - C @ x[t]
    return x, w, v

class Solver:
    def __init__(self):
        # Warm up JIT by compiling with small dummy data
        A = np.array([[1.0]])
        B = np.array([[1.0]])
        C = np.array([[1.0]])
        y = np.array([[1.0]])
        x0 = np.array([0.0])
        w = np.zeros(1)
        compute_obj_grad_numba(w, A, B, C, y, x0, 1.0, 1.0, 1, 1, 1, 1)
        reconstruct_solution(w, A, B, C, y, x0, 1, 1, 1, 1)
    
    def solve(self, problem, **kwargs):
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
        
        # Make copies to ensure contiguity
        A = np.ascontiguousarray(A)
        B = np.ascontiguousarray(B)
        C = np.ascontiguousarray(C)
        y = np.ascontiguousarray(y)
        x0 = np.ascontiguousarray(x0)
        
        def compute_obj_grad(w_flat):
            return compute_obj_grad_numba(w_flat, A, B, C, y, x0, tau, M, N, n, p, m)
        
        w0 = np.zeros(N * p)
        
        result = minimize(
            compute_obj_grad, w0,
            jac=True, method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-7}
        )
        
        x_opt, w_opt, v_opt = reconstruct_solution(result.x, A, B, C, y, x0, N, n, p, m)
        
        return {
            "x_hat": x_opt.tolist(),
            "w_hat": w_opt.tolist(),
            "v_hat": v_opt.tolist(),
        }