import numpy as np
from typing import Any
from numba import njit

@njit(cache=True, fastmath=True)
def solve_numba(X0, Y0, t0, t1, A, B):
    # Dormand-Prince 4(5) coefficients
    c2 = 1/5; c3 = 3/10; c4 = 4/5; c5 = 8/9; c6 = 1.0; c7 = 1.0
    
    a21 = 1/5
    a31 = 3/40; a32 = 9/40
    a41 = 44/45; a42 = -56/15; a43 = 32/9
    a51 = 19372/6561; a52 = -25360/2187; a53 = 64448/6561; a54 = -212/729
    a61 = 9017/3168; a62 = -355/33; a63 = 46732/5247; a64 = 49/176; a65 = -5103/18656
    a71 = 35/384; a72 = 0.0; a73 = 500/1113; a74 = 125/192; a75 = -2187/6784; a76 = 11/84
    
    b1 = 35/384; b3 = 500/1113; b4 = 125/192; b5 = -2187/6784; b6 = 11/84
    e1 = 71/57600; e3 = -71/16695; e4 = 71/1920; e5 = -17253/339200; e6 = 22/525; e7 = -1/40
    
    t = t0
    X = X0
    Y = Y0
    h = 1e-3
    rtol = 1e-8
    atol = 1e-8
    
    # Initial k1
    X2Y = X * X * Y
    k1_X = A + X2Y - (B + 1.0) * X
    k1_Y = B * X - X2Y

    while t < t1:
        if t + h > t1:
            h = t1 - t
            
        # k2
        X_tmp = X + h * a21 * k1_X
        Y_tmp = Y + h * a21 * k1_Y
        X2Y = X_tmp * X_tmp * Y_tmp
        k2_X = A + X2Y - (B + 1.0) * X_tmp
        k2_Y = B * X_tmp - X2Y
        
        # k3
        X_tmp = X + h * (a31 * k1_X + a32 * k2_X)
        Y_tmp = Y + h * (a31 * k1_Y + a32 * k2_Y)
        X2Y = X_tmp * X_tmp * Y_tmp
        k3_X = A + X2Y - (B + 1.0) * X_tmp
        k3_Y = B * X_tmp - X2Y
        
        # k4
        X_tmp = X + h * (a41 * k1_X + a42 * k2_X + a43 * k3_X)
        Y_tmp = Y + h * (a41 * k1_Y + a42 * k2_Y + a43 * k3_Y)
        X2Y = X_tmp * X_tmp * Y_tmp
        k4_X = A + X2Y - (B + 1.0) * X_tmp
        k4_Y = B * X_tmp - X2Y
        
        # k5
        X_tmp = X + h * (a51 * k1_X + a52 * k2_X + a53 * k3_X + a54 * k4_X)
        Y_tmp = Y + h * (a51 * k1_Y + a52 * k2_Y + a53 * k3_Y + a54 * k4_Y)
        X2Y = X_tmp * X_tmp * Y_tmp
        k5_X = A + X2Y - (B + 1.0) * X_tmp
        k5_Y = B * X_tmp - X2Y
        
        # k6
        X_tmp = X + h * (a61 * k1_X + a62 * k2_X + a63 * k3_X + a64 * k4_X + a65 * k5_X)
        Y_tmp = Y + h * (a61 * k1_Y + a62 * k2_Y + a63 * k3_Y + a64 * k4_Y + a65 * k5_Y)
        X2Y = X_tmp * X_tmp * Y_tmp
        k6_X = A + X2Y - (B + 1.0) * X_tmp
        k6_Y = B * X_tmp - X2Y
        
        # k7 (which is also the new k1 if step is accepted)
        X_new = X + h * (b1 * k1_X + b3 * k3_X + b4 * k4_X + b5 * k5_X + b6 * k6_X)
        Y_new = Y + h * (b1 * k1_Y + b3 * k3_Y + b4 * k4_Y + b5 * k5_Y + b6 * k6_Y)
        
        X2Y = X_new * X_new * Y_new
        k7_X = A + X2Y - (B + 1.0) * X_new
        k7_Y = B * X_new - X2Y
        
        err_X = h * (e1 * k1_X + e3 * k3_X + e4 * k4_X + e5 * k5_X + e6 * k6_X + e7 * k7_X)
        err_Y = h * (e1 * k1_Y + e3 * k3_Y + e4 * k4_Y + e5 * k5_Y + e6 * k6_Y + e7 * k7_Y)
        
        abs_X = X if X > 0 else -X
        abs_X_new = X_new if X_new > 0 else -X_new
        scale_X = atol + rtol * (abs_X if abs_X > abs_X_new else abs_X_new)
        
        abs_Y = Y if Y > 0 else -Y
        abs_Y_new = Y_new if Y_new > 0 else -Y_new
        scale_Y = atol + rtol * (abs_Y if abs_Y > abs_Y_new else abs_Y_new)
        
        err_X_norm = err_X / scale_X
        err_X_norm = err_X_norm if err_X_norm > 0 else -err_X_norm
        
        err_Y_norm = err_Y / scale_Y
        err_Y_norm = err_Y_norm if err_Y_norm > 0 else -err_Y_norm
        
        err_norm = err_X_norm if err_X_norm > err_Y_norm else err_Y_norm
        
        if err_norm <= 1.0:
            t += h
            X = X_new
            Y = Y_new
            k1_X = k7_X
            k1_Y = k7_Y
            if t >= t1:
                break
            
        if err_norm == 0:
            h *= 5.0
        else:
            factor = 0.9 * (err_norm ** -0.2)
            factor = min(max(factor, 0.2), 5.0)
            h *= factor
            
    return X, Y
            
    return X, Y

class Solver:
    def __init__(self):
        # Trigger compilation
        solve_numba(1.1, 3.2, 0.0, 1.0, 1.0, 3.0)

    def solve(self, problem: dict, **kwargs) -> Any:
        X0, Y0 = problem["y0"]
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        A = float(problem["params"]["A"])
        B = float(problem["params"]["B"])
        
        X_f, Y_f = solve_numba(X0, Y0, t0, t1, A, B)
        return np.array([X_f, Y_f])