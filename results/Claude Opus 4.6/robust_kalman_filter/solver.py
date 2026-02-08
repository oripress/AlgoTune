import numpy as np
from scipy.optimize import minimize
import numba as nb


@nb.njit(cache=True)
def _forward_pass(A, B, x0, w, N, n):
    x = np.empty((N + 1, n))
    for i in range(n):
        x[0, i] = x0[i]
    for t in range(N):
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i, j] * x[t, j]
            for j in range(w.shape[1]):
                s += B[i, j] * w[t, j]
            x[t + 1, i] = s
    return x


@nb.njit(cache=True)
def _obj_grad_numba(A, B, C, AT, BT, y, x0, w_flat, tau, M_val, N, n, m, p):
    w = w_flat.reshape(N, p)
    
    # Forward pass
    x = np.empty((N + 1, n))
    for i in range(n):
        x[0, i] = x0[i]
    for t in range(N):
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i, j] * x[t, j]
            for j in range(p):
                s += B[i, j] * w[t, j]
            x[t + 1, i] = s
    
    # Measurement noise: v[t] = y[t] - C @ x[t]
    v = np.empty((N, m))
    for t in range(N):
        for i in range(m):
            s = 0.0
            for j in range(n):
                s += C[i, j] * x[t, j]
            v[t, i] = y[t, i] - s
    
    # Norms and objective
    obj = 0.0
    for t in range(N):
        for j in range(p):
            obj += w[t, j] * w[t, j]
    
    v_norms = np.empty(N)
    for t in range(N):
        s = 0.0
        for j in range(m):
            s += v[t, j] * v[t, j]
        v_norms[t] = np.sqrt(s)
    
    for t in range(N):
        vn = v_norms[t]
        if vn <= M_val:
            obj += tau * vn * vn
        else:
            obj += tau * (2.0 * M_val * vn - M_val * M_val)
    
    # Gradient of Huber w.r.t. v
    # Then rhs[t] = -tau * C^T @ grad_v[t]
    rhs = np.zeros((N, n))
    for t in range(N):
        vn = v_norms[t]
        if vn < 1e-15:
            continue
        if vn <= M_val:
            scale = 2.0
        else:
            scale = 2.0 * M_val / vn
        # rhs[t] = -tau * C^T @ (scale * v[t])
        for i in range(n):
            s = 0.0
            for j in range(m):
                s += C[j, i] * v[t, j]  # C^T[i,j] = C[j,i]
            rhs[t, i] = -tau * scale * s
    
    # Backward pass
    grad_w = np.empty(N * p)
    p_next = np.zeros(n)
    tmp = np.empty(n)
    for t in range(N - 1, -1, -1):
        # grad_w[t] = 2*w[t] + B^T @ p_next
        for i in range(p):
            s = 0.0
            for j in range(n):
                s += BT[i, j] * p_next[j]
            grad_w[t * p + i] = 2.0 * w[t, i] + s
        # p_next = rhs[t] + A^T @ p_next
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += AT[i, j] * p_next[j]
            tmp[i] = rhs[t, i] + s
        for i in range(n):
            p_next[i] = tmp[i]
    
    return obj, grad_w


class Solver:
    def __init__(self):
        # Warm up numba JIT
        A = np.eye(1, dtype=np.float64)
        B = np.eye(1, dtype=np.float64)
        C = np.eye(1, dtype=np.float64)
        AT = np.eye(1, dtype=np.float64)
        BT = np.eye(1, dtype=np.float64)
        y = np.zeros((1, 1), dtype=np.float64)
        x0 = np.zeros(1, dtype=np.float64)
        w = np.zeros(1, dtype=np.float64)
        _obj_grad_numba(A, B, C, AT, BT, y, x0, w, 1.0, 1.0, 1, 1, 1, 1)

    def solve(self, problem, **kwargs):
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        C = np.array(problem["C"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        x0 = np.array(problem["x_initial"], dtype=np.float64)
        tau = float(problem["tau"])
        M_val = float(problem["M"])
        
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]
        
        AT = np.ascontiguousarray(A.T)
        BT = np.ascontiguousarray(B.T)
        CT = np.ascontiguousarray(C.T)
        
        def obj_grad(w_flat):
            return _obj_grad_numba(A, B, C, AT, BT, y, x0, w_flat, tau, M_val, N, n, m, p)
        
        w0 = np.zeros(N * p)
        res = minimize(obj_grad, w0, jac=True, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10})
        
        # Recover full solution
        w_opt = res.x.reshape(N, p)
        x_opt = np.empty((N + 1, n))
        x_opt[0] = x0
        for t in range(N):
            x_opt[t + 1] = A @ x_opt[t] + B @ w_opt[t]
        v_opt = y - x_opt[:N] @ CT
        
        return {
            "x_hat": x_opt.tolist(),
            "w_hat": w_opt.tolist(),
            "v_hat": v_opt.tolist(),
        }