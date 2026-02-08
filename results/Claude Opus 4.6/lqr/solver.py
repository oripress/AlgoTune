import numpy as np
import numba as nb
from scipy.linalg import cho_factor, cho_solve


@nb.njit(cache=True)
def _backward_forward(A, B, Q, R, P, T, x0):
    n = A.shape[0]
    m = B.shape[1]
    BT = B.T.copy()
    
    K_all = np.empty((T, m, n))
    S = P.copy()
    
    for t in range(T - 1, -1, -1):
        BtS = BT @ S
        M1 = R + BtS @ B
        M2 = BtS @ A
        K_all[t] = np.linalg.solve(M1, M2)
        Acl = A - B @ K_all[t]
        S_new = Q + K_all[t].T @ (R @ K_all[t]) + Acl.T @ (S @ Acl)
        for i in range(n):
            for j in range(i+1, n):
                v = 0.5 * (S_new[i,j] + S_new[j,i])
                S_new[i,j] = v
                S_new[j,i] = v
        
        # Check for convergence
        if t < T - 1:
            diff = 0.0
            nrm = 0.0
            for i in range(n):
                for j in range(n):
                    diff += (S_new[i,j] - S[i,j]) ** 2
                    nrm += S_new[i,j] ** 2
            if nrm > 0 and diff < nrm * 1e-28:
                for tt in range(t - 1, -1, -1):
                    for ii in range(m):
                        for jj in range(n):
                            K_all[tt, ii, jj] = K_all[t, ii, jj]
                S = S_new
                break
        S = S_new
    
    # Forward
    U = np.empty((T, m))
    x = x0.copy()
    for t in range(T):
        u = -(K_all[t] @ x)
        for j in range(m):
            U[t, j] = u[j, 0]
        x = A @ x + B @ u
    return U


@nb.njit(cache=True)
def _backward_forward_m1(A, b, Q, r, P, T, x0):
    """Specialized for m=1."""
    n = A.shape[0]
    bvec = b.ravel()
    
    K_all = np.empty((T, n))
    S = P.copy()
    
    for t in range(T - 1, -1, -1):
        Sb = S @ bvec
        denom = r + bvec @ Sb
        k = np.empty(n)
        for j in range(n):
            val = 0.0
            for i in range(n):
                val += Sb[i] * A[i, j]
            k[j] = val / denom
        K_all[t] = k
        
        Acl = A.copy()
        for i in range(n):
            for j in range(n):
                Acl[i, j] -= bvec[i] * k[j]
        
        S_new = Q + Acl.T @ (S @ Acl)
        for i in range(n):
            for j in range(n):
                S_new[i, j] += r * k[i] * k[j]
        
        for i in range(n):
            for j in range(i+1, n):
                v = 0.5 * (S_new[i,j] + S_new[j,i])
                S_new[i,j] = v
                S_new[j,i] = v
        
        if t < T - 1:
            diff = 0.0
            nrm = 0.0
            for i in range(n):
                for j in range(n):
                    diff += (S_new[i,j] - S[i,j]) ** 2
                    nrm += S_new[i,j] ** 2
            if nrm > 0 and diff < nrm * 1e-28:
                for tt in range(t - 1, -1, -1):
                    for jj in range(n):
                        K_all[tt, jj] = K_all[t, jj]
                S = S_new
                break
        S = S_new
    
    U = np.empty((T, 1))
    x = x0.ravel().copy()
    for t in range(T):
        u = 0.0
        for j in range(n):
            u -= K_all[t, j] * x[j]
        U[t, 0] = u
        x = A @ x + bvec * u
    return U


@nb.njit(cache=True)
def _solve_scalar(a, b, q, r, p, T, x0):
    K = np.empty(T)
    s = p
    for t in range(T-1, -1, -1):
        bs = b * s
        m1 = r + bs * b
        K[t] = bs * a / m1
        acl = a - b * K[t]
        s_new = q + K[t]*r*K[t] + acl*s*acl
        if t < T - 1 and abs(s) > 0 and abs(s_new - s) < abs(s) * 1e-14:
            for tt in range(t - 1, -1, -1):
                K[tt] = K[t]
            break
        s = s_new
    
    U = np.empty((T, 1))
    x = x0
    for t in range(T):
        u = -K[t] * x
        U[t, 0] = u
        x = a*x + b*u
    return U


def _solve_large(A, B, Q, R, P, T, x0):
    """For large matrices, use scipy."""
    n, m = B.shape
    BT = B.T
    K_all = np.empty((T, m, n))
    S = P.copy()
    
    for t in range(T - 1, -1, -1):
        BtS = BT @ S
        M1 = R + BtS @ B
        M2 = BtS @ A
        try:
            cf = cho_factor(M1)
            K_all[t] = cho_solve(cf, M2)
        except Exception:
            K_all[t] = np.linalg.solve(M1, M2)
        Acl = A - B @ K_all[t]
        S = Q + K_all[t].T @ R @ K_all[t] + Acl.T @ S @ Acl
        S = (S + S.T) * 0.5
    
    U = np.empty((T, m))
    x = x0.reshape(n, 1)
    for t in range(T):
        u = -K_all[t] @ x
        U[t] = u.ravel()
        x = A @ x + B @ u
    return U


class Solver:
    def __init__(self):
        _backward_forward(np.eye(2), np.ones((2,2)), np.eye(2), np.eye(2), np.eye(2), 1, np.ones((2,1)))
        _backward_forward_m1(np.eye(2), np.ones((2,1)), np.eye(2), 1.0, np.eye(2), 1, np.ones((2,1)))
        _solve_scalar(1.0, 1.0, 1.0, 1.0, 1.0, 1, 1.0)

    def solve(self, problem, **kwargs):
        T = int(problem["T"])
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        Q = np.asarray(problem["Q"], dtype=np.float64)
        R = np.asarray(problem["R"], dtype=np.float64)
        P = np.asarray(problem["P"], dtype=np.float64)
        x0 = np.asarray(problem["x0"], dtype=np.float64)
        
        n, m = B.shape
        
        if n == 1 and m == 1:
            return {"U": _solve_scalar(A[0,0], B[0,0], Q[0,0], R[0,0], P[0,0], T, x0.ravel()[0])}
        
        # For larger matrices, scipy may be faster
        if n > 10 and m > 3:
            return {"U": _solve_large(A, B, Q, R, P, T, x0)}
        
        if m == 1:
            return {"U": _backward_forward_m1(A, B, Q, R[0,0], P, T, x0.reshape(n,1))}
        
        return {"U": _backward_forward(A, B, Q, R, P, T, x0.reshape(n,1))}