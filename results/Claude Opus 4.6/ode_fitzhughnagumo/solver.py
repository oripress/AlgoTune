import numpy as np
from numba import njit
from typing import Any

@njit(cache=True, fastmath=True)
def rk45_solve(t0, t1, v0, w0, a, b, c, I, rtol, atol):
    """Adaptive RK45 (Dormand-Prince) FSAL solver for FitzHugh-Nagumo system."""
    b21 = 1.0/5.0
    b31 = 3.0/40.0; b32 = 9.0/40.0
    b41 = 44.0/45.0; b42 = -56.0/15.0; b43 = 32.0/9.0
    b51 = 19372.0/6561.0; b52 = -25360.0/2187.0; b53 = 64448.0/6561.0; b54 = -212.0/729.0
    b61 = 9017.0/3168.0; b62 = -355.0/33.0; b63 = 46732.0/5247.0; b64 = 49.0/176.0; b65 = -5103.0/18656.0
    
    c1 = 35.0/384.0; c3 = 500.0/1113.0; c4 = 125.0/192.0; c5 = -2187.0/6784.0; c6 = 11.0/84.0
    e1 = 71.0/57600.0; e3 = -71.0/16695.0; e4 = 71.0/1920.0; e5 = -17253.0/339200.0; e6 = 22.0/525.0; e7 = -1.0/40.0
    
    t = t0; v = v0; w = w0
    
    # Initial k1
    k1v = v - v*v*v/3.0 - w + I
    k1w = a * (b * v - c * w)
    
    f_norm = (k1v*k1v + k1w*k1w)**0.5
    h = min(0.01 * ((v*v + w*w)**0.5 + 1e-6) / (f_norm + 1e-15), (t1 - t0) * 0.1)
    if h < 1e-14:
        h = (t1 - t0) * 0.001
    
    have_k1 = True
    
    while t < t1:
        if t + h > t1:
            h = t1 - t
        if h < 1e-15:
            break
        
        if not have_k1:
            k1v = v - v*v*v/3.0 - w + I
            k1w = a * (b * v - c * w)
        
        v2 = v + h*b21*k1v; w2 = w + h*b21*k1w
        k2v = v2 - v2*v2*v2/3.0 - w2 + I; k2w = a*(b*v2 - c*w2)
        
        v3 = v + h*(b31*k1v + b32*k2v); w3 = w + h*(b31*k1w + b32*k2w)
        k3v = v3 - v3*v3*v3/3.0 - w3 + I; k3w = a*(b*v3 - c*w3)
        
        v4 = v + h*(b41*k1v + b42*k2v + b43*k3v); w4 = w + h*(b41*k1w + b42*k2w + b43*k3w)
        k4v = v4 - v4*v4*v4/3.0 - w4 + I; k4w = a*(b*v4 - c*w4)
        
        v5 = v + h*(b51*k1v + b52*k2v + b53*k3v + b54*k4v); w5 = w + h*(b51*k1w + b52*k2w + b53*k3w + b54*k4w)
        k5v = v5 - v5*v5*v5/3.0 - w5 + I; k5w = a*(b*v5 - c*w5)
        
        v6 = v + h*(b61*k1v + b62*k2v + b63*k3v + b64*k4v + b65*k5v); w6 = w + h*(b61*k1w + b62*k2w + b63*k3w + b64*k4w + b65*k5w)
        k6v = v6 - v6*v6*v6/3.0 - w6 + I; k6w = a*(b*v6 - c*w6)
        
        v_new = v + h*(c1*k1v + c3*k3v + c4*k4v + c5*k5v + c6*k6v)
        w_new = w + h*(c1*k1w + c3*k3w + c4*k4w + c5*k5w + c6*k6w)
        
        k7v = v_new - v_new*v_new*v_new/3.0 - w_new + I
        k7w = a*(b*v_new - c*w_new)
        
        err_v = h*(e1*k1v + e3*k3v + e4*k4v + e5*k5v + e6*k6v + e7*k7v)
        err_w = h*(e1*k1w + e3*k3w + e4*k4w + e5*k5w + e6*k6w + e7*k7w)
        
        sc_v = atol + rtol*max(abs(v), abs(v_new))
        sc_w = atol + rtol*max(abs(w), abs(w_new))
        err_norm = ((err_v/sc_v)**2 + (err_w/sc_w)**2)**0.5 * 0.7071067811865476
        
        if err_norm <= 1.0:
            t += h; v = v_new; w = w_new
            k1v = k7v; k1w = k7w  # FSAL
            have_k1 = True
            if err_norm < 1e-15:
                h *= 5.0
            else:
                h *= min(5.0, max(0.2, 0.9 * err_norm**(-0.2)))
        else:
            have_k1 = False
            h *= max(0.2, 0.9 * err_norm**(-0.2))
    
    return v, w

# Warm up numba
_warmup = rk45_solve(0.0, 1.0, -1.0, -0.5, 0.08, 0.8, 0.7, 0.5, 1e-8, 1e-8)

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        t0 = problem["t0"]
        t1 = problem["t1"]
        y0 = problem["y0"]
        params = problem["params"]
        v, w = rk45_solve(t0, t1, y0[0], y0[1], params["a"], params["b"], params["c"], params["I"], 1e-8, 1e-8)
        return [v, w]