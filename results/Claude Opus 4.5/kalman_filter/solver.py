import numpy as np
from numba import njit

@njit(cache=True)
def backward_pass(A, B, CtC, Cty, N, n, p):
    K_arr = np.zeros((N, p, n))
    k_arr = np.zeros((N, p))
    P = np.zeros((n, n))
    q = np.zeros(n)
    
    for t in range(N - 1, -1, -1):
        BtP = B.T @ P
        BtPB = BtP @ B
        M = np.eye(p) + BtPB
        M_inv = np.linalg.inv(M)
        BtPA = BtP @ A
        Btq = B.T @ q
        K_arr[t] = M_inv @ BtPA
        k_arr[t] = M_inv @ Btq
        
        AtP = A.T @ P
        AtPB = AtP @ B
        P = CtC + AtP @ A - AtPB @ K_arr[t]
        q = -Cty[t] + A.T @ q - AtPB @ k_arr[t]
    
    return K_arr, k_arr

@njit(cache=True)
def forward_pass(A, B, C, y, x0, K_arr, k_arr, N, n, p, m):
    x_hat = np.zeros((N + 1, n))
    w_hat = np.zeros((N, p))
    v_hat = np.zeros((N, m))
    
    x_hat[0] = x0
    for t in range(N):
        w_hat[t] = -K_arr[t] @ x_hat[t] - k_arr[t]
        x_hat[t + 1] = A @ x_hat[t] + B @ w_hat[t]
        v_hat[t] = y[t] - C @ x_hat[t]
    
    return x_hat, w_hat, v_hat

class Solver:
    def solve(self, problem, **kwargs):
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        C = np.array(problem["C"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        x0 = np.array(problem["x_initial"], dtype=np.float64)
        tau = float(problem["tau"])
        
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]
        
        if N == 0:
            return {
                "x_hat": [x0.tolist()],
                "w_hat": [],
                "v_hat": []
            }
        
        CtC = tau * (C.T @ C)
        Cty = tau * (y @ C)  # Shape (N, n)
        
        K_arr, k_arr = backward_pass(A, B, CtC, Cty, N, n, p)
        x_hat, w_hat, v_hat = forward_pass(A, B, C, y, x0, K_arr, k_arr, N, n, p, m)
        
        return {
            "x_hat": x_hat.tolist(),
            "w_hat": w_hat.tolist(),
            "v_hat": v_hat.tolist()
        }