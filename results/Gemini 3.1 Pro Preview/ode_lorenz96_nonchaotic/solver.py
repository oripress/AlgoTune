import numpy as np
from numba import njit

@njit(fastmath=True, inline='always')
def lorenz96_numba(x, F, dxdt):
    N = len(x)
    dxdt[0] = (x[1] - x[N - 2]) * x[N - 1] - x[0] + F
    dxdt[1] = (x[2] - x[N - 1]) * x[0] - x[1] + F
    for i in range(2, N - 1):
        dxdt[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i] + F
    dxdt[N - 1] = (x[0] - x[N - 3]) * x[N - 2] - x[N - 1] + F

@njit(fastmath=True, inline='always')
def rk45_step(y, h, F, f0, k2, k3, k4, k5, k6, k7, y_tmp, error):
    N = len(y)
    
    h_1_5 = h * 0.2
    for i in range(N):
        y_tmp[i] = y[i] + h_1_5 * f0[i]
    lorenz96_numba(y_tmp, F, k2)
    
    h_3_40 = h * 0.075
    h_9_40 = h * 0.225
    for i in range(N):
        y_tmp[i] = y[i] + h_3_40 * f0[i] + h_9_40 * k2[i]
    lorenz96_numba(y_tmp, F, k3)
    
    h_44_45 = h * (44/45)
    h_56_15 = h * (-56/15)
    h_32_9 = h * (32/9)
    for i in range(N):
        y_tmp[i] = y[i] + h_44_45 * f0[i] + h_56_15 * k2[i] + h_32_9 * k3[i]
    lorenz96_numba(y_tmp, F, k4)
    
    h_19372_6561 = h * (19372/6561)
    h_25360_2187 = h * (-25360/2187)
    h_64448_6561 = h * (64448/6561)
    h_212_729 = h * (-212/729)
    for i in range(N):
        y_tmp[i] = y[i] + h_19372_6561 * f0[i] + h_25360_2187 * k2[i] + h_64448_6561 * k3[i] + h_212_729 * k4[i]
    lorenz96_numba(y_tmp, F, k5)
    
    h_9017_3168 = h * (9017/3168)
    h_355_33 = h * (-355/33)
    h_46732_5247 = h * (46732/5247)
    h_49_176 = h * (49/176)
    h_5103_18656 = h * (-5103/18656)
    for i in range(N):
        y_tmp[i] = y[i] + h_9017_3168 * f0[i] + h_355_33 * k2[i] + h_46732_5247 * k3[i] + h_49_176 * k4[i] + h_5103_18656 * k5[i]
    lorenz96_numba(y_tmp, F, k6)
    
    h_35_384 = h * (35/384)
    h_500_1113 = h * (500/1113)
    h_125_192 = h * (125/192)
    h_2187_6784 = h * (-2187/6784)
    h_11_84 = h * (11/84)
    for i in range(N):
        y_tmp[i] = y[i] + h_35_384 * f0[i] + h_500_1113 * k3[i] + h_125_192 * k4[i] + h_2187_6784 * k5[i] + h_11_84 * k6[i]
    lorenz96_numba(y_tmp, F, k7)
    
    e_1 = h * (-71/57600)
    e_3 = h * (71/16695)
    e_4 = h * (-71/1920)
    e_5 = h * (17253/339200)
    e_6 = h * (-22/525)
    e_7 = h * (1/40)
    for i in range(N):
        error[i] = e_1 * f0[i] + e_3 * k3[i] + e_4 * k4[i] + e_5 * k5[i] + e_6 * k6[i] + e_7 * k7[i]
@njit(fastmath=True)
def rk45_solve(t0, t1, y0, F, rtol, atol):
    N = len(y0)
    t = t0
    y = y0.copy()
    f0 = np.empty(N)
    k2 = np.empty(N)
    k3 = np.empty(N)
    k4 = np.empty(N)
    k5 = np.empty(N)
    k6 = np.empty(N)
    k7 = np.empty(N)
    y_tmp = np.empty(N)
    error = np.empty(N)
    
    lorenz96_numba(y, F, f0)
    
    d0_sq = 0.0
    d1_sq = 0.0
    for i in range(N):
        scale_i = atol + abs(y[i]) * rtol
        d0_sq += (y[i] / scale_i)**2
        d1_sq += (f0[i] / scale_i)**2
    d0 = np.sqrt(d0_sq / N)
    d1 = np.sqrt(d1_sq / N)
    
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    
    for i in range(N):
        y_tmp[i] = y[i] + h0 * f0[i]
    lorenz96_numba(y_tmp, F, k2)
    
    d2_sq = 0.0
    for i in range(N):
        scale_i = atol + abs(y[i]) * rtol
        d2_sq += ((k2[i] - f0[i]) / scale_i)**2
    d2 = np.sqrt(d2_sq / N) / h0
    if max(d1, d2) <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** 0.2
    
    h = min(100 * h0, h1)
    
    step_rejected = False
    
    while t < t1:
        if t + h > t1:
            h = t1 - t
            
        rk45_step(y, h, F, f0, k2, k3, k4, k5, k6, k7, y_tmp, error)
        
        error_norm = 0.0
        error_norm_sq = 0.0
        for i in range(N):
            scale_i = atol + max(abs(y[i]), abs(y_tmp[i])) * rtol
            error_norm_sq += (error[i] / scale_i)**2
        error_norm = np.sqrt(error_norm_sq / N)
        if error_norm < 1:
            t += h
            for i in range(N):
                y[i] = y_tmp[i]
                f0[i] = k7[i]
            
            if error_norm == 0:
                factor = 10.0
            else:
                factor = min(10.0, 0.9 * error_norm ** -0.2)
                
            if step_rejected:
                factor = min(1.0, factor)
                
            h *= factor
            step_rejected = False
        else:
            factor = max(0.2, 0.9 * error_norm ** -0.2)
            h *= factor
            step_rejected = True
            
    return y

class Solver:
    def __init__(self):
        # Trigger Numba compilation
        dummy_y0 = np.array([1.0, 2.0, 3.0, 4.0])
        rk45_solve(0.0, 1.0, dummy_y0, 2.0, 1e-8, 1e-8)

    def solve(self, problem: dict, **kwargs):
        y0 = np.array(problem["y0"])
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        F = float(problem["F"])
        
        sol = rk45_solve(t0, t1, y0, F, 1e-8, 1e-8)
        return sol.tolist()