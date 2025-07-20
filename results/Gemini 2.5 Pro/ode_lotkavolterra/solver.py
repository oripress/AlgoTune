from typing import Any
import numpy as np
from numba import jit

# --- Butcher Tableau for the Dormand-Prince 5(4) method (DOPRI5) ---
C_NODES = np.array([0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0], dtype=np.float64)
A_MATRIX = np.array([
    [1/5, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3/40, 9/40, 0.0, 0.0, 0.0, 0.0],
    [44/45, -56/15, 32/9, 0.0, 0.0, 0.0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0.0, 0.0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0.0],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
], dtype=np.float64)
B_COEFFS = np.array([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0], dtype=np.float64)
E_COEFFS = np.array([71/57600, 0.0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40], dtype=np.float64)

@jit(nopython=True)
def lotka_volterra_rhs_inplace(out, y, t, alpha, beta, delta, gamma):
    """
    The RHS of the Lotka-Volterra ODEs, calculated in-place to avoid
    memory allocation in the hot loop.
    """
    x, y_pred = y
    out[0] = alpha * x - beta * x * y_pred
    out[1] = delta * x * y_pred - gamma * y_pred

@jit(nopython=True)
def solve_ode_dopri5(t0, t1, y0, params, rtol, atol):
    """
    A maximally optimized, JIT-compiled, adaptive DOPRI5 solver.
    """
    alpha, beta, delta, gamma = params
    t = t0
    y = y0.copy()
    
    k = np.zeros((7, len(y0)), dtype=np.float64)
    y_stage = np.zeros_like(y0)
    y_new = np.zeros_like(y0)

    h = (t1 - t0) * 0.01
    
    lotka_volterra_rhs_inplace(k[0], y, t, alpha, beta, delta, gamma)
    
    while t < t1:
        if t + h > t1:
            h = t1 - t

        for i in range(1, 7):
            y_stage[:] = y
            for j in range(i):
                y_stage += h * A_MATRIX[i-1, j] * k[j]
            lotka_volterra_rhs_inplace(k[i], y_stage, t + C_NODES[i] * h, alpha, beta, delta, gamma)

        y_new[:] = y + h * np.dot(k.T, B_COEFFS)
        err_est_vec = h * np.dot(k.T, E_COEFFS)
        
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.sqrt(np.mean((err_est_vec / (scale + 1e-16))**2))

        if err_norm <= 1.0:
            t += h
            y[:] = y_new
            k[0, :] = k[6, :]
        
        if err_norm > 1e-16:
            h = h * min(5.0, max(0.2, 0.9 * (err_norm)**-0.2))
        else:
            h = h * 5.0

    return y

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        p = problem["params"]
        params_tuple = (p["alpha"], p["beta"], p["delta"], p["gamma"])

        final_y = solve_ode_dopri5(t0, t1, y0, params_tuple, rtol=1e-10, atol=1e-10)
        return final_y.tolist()