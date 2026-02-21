import numpy as np
from scipy import signal, linalg
import numba

@numba.njit(cache=True, fastmath=True)
def fast_expm(A):
    n = A.shape[0]
    # Infinity norm
    A_norm = 0.0
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            row_sum += abs(A[i, j])
        if row_sum > A_norm:
            A_norm = row_sum
            
    # Scaling
    if A_norm > 0:
        s = int(np.ceil(np.log2(A_norm)))
        s = max(0, s + 1)
    else:
        s = 0
        
    A_scaled = A / (2.0 ** s)
    
    # Pade approximant degree 7
    c = [1.0, 0.5, 0.11666666666666667, 0.016666666666666666, 0.0015873015873015873, 0.00010582010582010583, 4.712031439648392e-06, 1.3462946970423977e-07]
    
    A2 = A_scaled @ A_scaled
    A4 = A2 @ A2
    A6 = A4 @ A2
    
    U = c[0]*np.eye(n) + c[2]*A2 + c[4]*A4 + c[6]*A6
    V = c[1]*np.eye(n) + c[3]*A2 + c[5]*A4 + c[7]*A6
    
    U = U + A_scaled @ V
    V = U - 2 * A_scaled @ V
    
    # Inverse of V
    F = np.linalg.solve(V, U)
    
    # Squaring
    for _ in range(s):
        F = F @ F
        
    return F

@numba.njit(cache=True, fastmath=True)
def simulate_loop(n_steps, n_states, Ad, Bd0, Bd1, U, C, D):
    xout = np.zeros((n_steps, n_states), dtype=np.float64)
    yout = np.zeros(n_steps, dtype=np.float64)
    
    # Initial state is 0
    yout[0] = U[0] * D[0, 0]
    
    for i in range(1, n_steps):
        for j in range(n_states):
            val = 0.0
            for k in range(n_states):
                val += xout[i-1, k] * Ad[k, j]
            val += U[i-1] * Bd0[0, j] + U[i] * Bd1[0, j]
            xout[i, j] = val
            
        y_val = 0.0
        for j in range(n_states):
            y_val += xout[i, j] * C[0, j]
        y_val += U[i] * D[0, 0]
        yout[i] = y_val
        
    return yout

class Solver:
    def __init__(self):
        # Pre-compile the numba function
        simulate_loop(2, 2, np.eye(2), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros(2), np.zeros((1, 2)), np.zeros((1, 1)))
        fast_expm(np.eye(2))

    def solve(self, problem: dict, **kwargs) -> dict:
        num = np.array(problem["num"], dtype=float)
        den = np.array(problem["den"], dtype=float)
        u = np.array(problem["u"], dtype=float)
        t = np.array(problem["t"], dtype=float)
        
        # Fast tf2ss
        num = np.atleast_1d(num)
        den = np.atleast_1d(den)
        
        # Normalize denominator
        if den[0] != 1.0:
            num = num / den[0]
            den = den / den[0]
            
        n_states = max(len(num), len(den)) - 1
        
        if n_states == 0:
            A = np.zeros((0, 0))
            B = np.zeros((0, 1))
            C = np.zeros((1, 0))
            D = np.array([[num[0]]])
        else:
            # Pad num and den to have length n_states + 1
            num_padded = np.zeros(n_states + 1)
            num_padded[-len(num):] = num
            den_padded = np.zeros(n_states + 1)
            den_padded[-len(den):] = den
            
            A = np.zeros((n_states, n_states))
            A[:-1, 1:] = np.eye(n_states - 1)
            A[-1, :] = -den_padded[1:][::-1]
            
            B = np.zeros((n_states, 1))
            B[-1, 0] = 1.0
            
            D = np.array([[num_padded[0]]])
            
            C = np.zeros((1, n_states))
            C[0, :] = num_padded[1:][::-1] - num_padded[0] * den_padded[1:][::-1]
            
        n_inputs = B.shape[1]
        n_steps = t.size
        
        if n_steps <= 1:
            return {"yout": [0.0] * n_steps}
            
        dt = t[1] - t[0]
        
        M = np.zeros((n_states + 2 * n_inputs, n_states + 2 * n_inputs))
        M[:n_states, :n_states] = A * dt
        M[:n_states, n_states:n_states+n_inputs] = B * dt
        M[n_states:n_states+n_inputs, n_states+n_inputs:] = np.eye(n_inputs)
        
        expMT = fast_expm(M.T)
        Ad = np.ascontiguousarray(expMT[:n_states, :n_states])
        Bd1 = np.ascontiguousarray(expMT[n_states+n_inputs:, :n_states])
        Bd0 = np.ascontiguousarray(expMT[n_states:n_states + n_inputs, :n_states] - Bd1)
        
        C = np.ascontiguousarray(C)
        D = np.ascontiguousarray(D)
        
        yout = simulate_loop(n_steps, n_states, Ad, Bd0, Bd1, u, C, D)
        
        return {"yout": yout.tolist()}