import numpy as np
from numba import njit
from typing import Any

@njit(cache=True, fastmath=True)
def _solve_brusselator(t0, t1, X0, Y0, A, B, rtol, atol):
    # Dormand-Prince RK45 coefficients
    a21 = 1.0/5.0
    a31 = 3.0/40.0; a32 = 9.0/40.0
    a41 = 44.0/45.0; a42 = -56.0/15.0; a43 = 32.0/9.0
    a51 = 19372.0/6561.0; a52 = -25360.0/2187.0; a53 = 64448.0/6561.0; a54 = -212.0/729.0
    a61 = 9017.0/3168.0; a62 = -355.0/33.0; a63 = 46732.0/5247.0; a64 = 49.0/176.0; a65 = -5103.0/18656.0
    
    b1 = 35.0/384.0; b3 = 500.0/1113.0; b4 = 125.0/192.0; b5 = -2187.0/6784.0; b6 = 11.0/84.0
    
    # Error coefficients (b - b*)
    e1 = 71.0/57600.0; e3 = -71.0/16695.0; e4 = 71.0/1920.0; e5 = -17253.0/339200.0; e6 = 22.0/525.0; e7 = -1.0/40.0
    
    Bp1 = B + 1.0
    
    t = t0
    X = X0
    Y = Y0
    
    # Initial step size estimate - compute f(t0, y0)
    k1X = A + X*X*Y - Bp1*X
    k1Y = B*X - X*X*Y
    
    d0 = np.sqrt(X*X + Y*Y)
    d1 = np.sqrt(k1X*k1X + k1Y*k1Y)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    
    h = min(h0, t1 - t0)
    
    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0
    
    have_k1 = True  # We already computed k1
    
    max_steps = 10000000
    
    for _ in range(max_steps):
        if t >= t1:
            break
        
        if t + h > t1:
            h = t1 - t
        
        if h < 1e-14:
            break
        
        # Stage 1 - use FSAL
        if not have_k1:
            k1X = A + X*X*Y - Bp1*X
            k1Y = B*X - X*X*Y
        
        # Stage 2
        Xt = X + h*a21*k1X
        Yt = Y + h*a21*k1Y
        k2X = A + Xt*Xt*Yt - Bp1*Xt
        k2Y = B*Xt - Xt*Xt*Yt
        
        # Stage 3
        Xt = X + h*(a31*k1X + a32*k2X)
        Yt = Y + h*(a31*k1Y + a32*k2Y)
        k3X = A + Xt*Xt*Yt - Bp1*Xt
        k3Y = B*Xt - Xt*Xt*Yt
        
        # Stage 4
        Xt = X + h*(a41*k1X + a42*k2X + a43*k3X)
        Yt = Y + h*(a41*k1Y + a42*k2Y + a43*k3Y)
        k4X = A + Xt*Xt*Yt - Bp1*Xt
        k4Y = B*Xt - Xt*Xt*Yt
        
        # Stage 5
        Xt = X + h*(a51*k1X + a52*k2X + a53*k3X + a54*k4X)
        Yt = Y + h*(a51*k1Y + a52*k2Y + a53*k3Y + a54*k4Y)
        k5X = A + Xt*Xt*Yt - Bp1*Xt
        k5Y = B*Xt - Xt*Xt*Yt
        
        # Stage 6
        Xt = X + h*(a61*k1X + a62*k2X + a63*k3X + a64*k4X + a65*k5X)
        Yt = Y + h*(a61*k1Y + a62*k2Y + a63*k3Y + a64*k4Y + a65*k5Y)
        k6X = A + Xt*Xt*Yt - Bp1*Xt
        k6Y = B*Xt - Xt*Xt*Yt
        
        # 5th order solution
        Xnew = X + h*(b1*k1X + b3*k3X + b4*k4X + b5*k5X + b6*k6X)
        Ynew = Y + h*(b1*k1Y + b3*k3Y + b4*k4Y + b5*k5Y + b6*k6Y)
        
        # Stage 7 (FSAL - this becomes k1 for next step)
        k7X = A + Xnew*Xnew*Ynew - Bp1*Xnew
        k7Y = B*Xnew - Xnew*Xnew*Ynew
        
        # Error estimate
        errX = h*(e1*k1X + e3*k3X + e4*k4X + e5*k5X + e6*k6X + e7*k7X)
        errY = h*(e1*k1Y + e3*k3Y + e4*k4Y + e5*k5Y + e6*k6Y + e7*k7Y)
        
        # Compute error norm
        scX = atol + rtol * max(abs(X), abs(Xnew))
        scY = atol + rtol * max(abs(Y), abs(Ynew))
        err_norm = np.sqrt(0.5 * ((errX/scX)**2 + (errY/scY)**2))
        
        if err_norm <= 1.0:
            # Accept step
            t = t + h
            X = Xnew
            Y = Ynew
            # FSAL: k7 becomes k1 for next step
            k1X = k7X
            k1Y = k7Y
            have_k1 = True
            
            # Compute new step size
            if err_norm == 0.0:
                factor = max_factor
            else:
                factor = min(max_factor, max(min_factor, safety * err_norm**(-0.2)))
            h = h * factor
        else:
            # Reject step - need to recompute k1 next time
            have_k1 = True  # k1 is still valid since X,Y didn't change
            factor = max(min_factor, safety * err_norm**(-0.2))
            h = h * factor
    
    return X, Y

# Warm up numba compilation at import time
_dummy = _solve_brusselator(0.0, 0.1, 1.0, 1.0, 1.0, 3.0, 1e-8, 1e-8)

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        t0 = problem["t0"]
        t1 = problem["t1"]
        y0 = problem["y0"]
        A = problem["params"]["A"]
        B = problem["params"]["B"]
        
        X, Y = _solve_brusselator(t0, t1, y0[0], y0[1], A, B, 1e-8, 1e-8)
        
        return np.array([X, Y])