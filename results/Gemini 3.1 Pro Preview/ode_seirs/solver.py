from typing import Any
import numpy as np
from numba import njit

A10 = 1/5
A20 = 3/40; A21 = 9/40
A30 = 44/45; A31 = -56/15; A32 = 32/9
A40 = 19372/6561; A41 = -25360/2187; A42 = 64448/6561; A43 = -212/729
A50 = 9017/3168; A51 = -355/33; A52 = 46732/5247; A53 = 49/176; A54 = -5103/18656

B0 = 35/384; B1 = 0.0; B2 = 500/1113; B3 = 125/192; B4 = -2187/6784; B5 = 11/84
C1 = 1/5; C2 = 3/10; C3 = 4/5; C4 = 8/9; C5 = 1.0
E0 = -71/57600; E1 = 0.0; E2 = 71/16695; E3 = -71/1920; E4 = 17253/339200; E5 = -22/525; E6 = 1/40

@njit(fastmath=True, inline='always')
def seirs_numba(t, y, out, beta, sigma, gamma, omega):
    S, E_var, I, R = y[0], y[1], y[2], y[3]
    out[0] = -beta * S * I + omega * R
    out[1] = beta * S * I - sigma * E_var
    out[2] = sigma * E_var - gamma * I
    out[3] = gamma * I - omega * R

@njit(fastmath=True)
def rk45_solve(t0, y0, t1, rtol, atol, beta, sigma, gamma, omega):
    t = t0
    y = y0.copy()
    
    K = np.empty((7, 4))
    y_tmp = np.empty(4)
    y_new = np.empty(4)
    
    seirs_numba(t, y, K[0], beta, sigma, gamma, omega)
    
    d0 = 0.0
    d1 = 0.0
    for i in range(4):
        sc = atol + abs(y[i]) * rtol
        d0 += (y[i] / sc)**2
        d1 += (K[0][i] / sc)**2
    d0 = np.sqrt(d0 / 4)
    d1 = np.sqrt(d1 / 4)
    
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
        
    for i in range(4):
        y_tmp[i] = y[i] + h0 * K[0][i]
        
    seirs_numba(t + h0, y_tmp, K[1], beta, sigma, gamma, omega)
    
    d2 = 0.0
    for i in range(4):
        sc = atol + abs(y[i]) * rtol
        d2 += ((K[1][i] - K[0][i]) / sc)**2
    d2 = np.sqrt(d2 / 4) / h0
    
    max_d1_d2 = max(d1, d2)
    if max_d1_d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max_d1_d2) ** 0.2
        
    h = min(100 * h0, h1)
    
    while t < t1:
        if t + h > t1:
            h = t1 - t
            
        for i in range(4):
            y_tmp[i] = y[i] + h * (A10*K[0][i])
        seirs_numba(t + C1*h, y_tmp, K[1], beta, sigma, gamma, omega)
        
        for i in range(4):
            y_tmp[i] = y[i] + h * (A20*K[0][i] + A21*K[1][i])
        seirs_numba(t + C2*h, y_tmp, K[2], beta, sigma, gamma, omega)
        
        for i in range(4):
            y_tmp[i] = y[i] + h * (A30*K[0][i] + A31*K[1][i] + A32*K[2][i])
        seirs_numba(t + C3*h, y_tmp, K[3], beta, sigma, gamma, omega)
        
        for i in range(4):
            y_tmp[i] = y[i] + h * (A40*K[0][i] + A41*K[1][i] + A42*K[2][i] + A43*K[3][i])
        seirs_numba(t + C4*h, y_tmp, K[4], beta, sigma, gamma, omega)
        
        for i in range(4):
            y_tmp[i] = y[i] + h * (A50*K[0][i] + A51*K[1][i] + A52*K[2][i] + A53*K[3][i] + A54*K[4][i])
        seirs_numba(t + C5*h, y_tmp, K[5], beta, sigma, gamma, omega)
        
        for i in range(4):
            y_new[i] = y[i] + h * (B0*K[0][i] + B1*K[1][i] + B2*K[2][i] + B3*K[3][i] + B4*K[4][i] + B5*K[5][i])
        seirs_numba(t + h, y_new, K[6], beta, sigma, gamma, omega)
        
        err_norm_sq = 0.0
        for i in range(4):
            err = h * (E0*K[0][i] + E1*K[1][i] + E2*K[2][i] + E3*K[3][i] + E4*K[4][i] + E5*K[5][i] + E6*K[6][i])
            sc = atol + max(abs(y[i]), abs(y_new[i])) * rtol
            err_norm_sq += (err / sc)**2
        err_norm = np.sqrt(err_norm_sq / 4)
        
        if err_norm < 1.0:
            t += h
            for i in range(4):
                y[i] = y_new[i]
                K[0][i] = K[6][i]
            
        if err_norm == 0:
            factor = 10.0
        else:
            factor = 0.9 * err_norm ** -0.2
            
        factor = max(0.2, min(10.0, factor))
        h *= factor
        
    return y

class Solver:
    def __init__(self):
        rk45_solve(0.0, np.array([0.89, 0.01, 0.005, 0.095]), 1.0, 1e-10, 1e-10, 0.35, 0.2, 0.1, 0.002)

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        beta = params["beta"]
        sigma = params["sigma"]
        gamma = params["gamma"]
        omega = params["omega"]

        y_final = rk45_solve(t0, y0, t1, 1e-10, 1e-10, beta, sigma, gamma, omega)
        return y_final.tolist()