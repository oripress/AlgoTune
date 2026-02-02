from typing import Any
import numpy as np
from numba import njit

@njit(fastmath=True)
def solve_jit_scalar_opt(v0, w0, t0, t1, a, b, c, I):
    t = t0
    v = v0
    w = w0
    
    # Initial step size guess - slightly aggressive
    # Initial step size guess - conservative to avoid initial error
    h = 1e-3
    
    # Strict tolerances to match reference
    rtol = 1e-8
    atol = 1e-8
    
    # Constants
    one_third = 1.0/3.0
    
    # RK45 Coefficients (Dormand-Prince)
    a21=1/5
    a31=3/40; a32=9/40
    a41=44/45; a42=-56/15; a43=32/9
    a51=19372/6561; a52=-25360/2187; a53=64448/6561; a54=-212/729
    a61=9017/3168; a62=-355/33; a63=46732/5247; a64=49/176; a65=-5103/18656
    
    b1=35/384; b3=500/1113; b4=125/192; b5=-2187/6784; b6=11/84
    
    e1=71/57600; e3=-71/16695; e4=71/1920; e5=-17253/339200; e6=22/525; e7=-1/40
    
    # Initial derivatives
    k1_v = v - v*v*v*one_third - w + I
    k1_w = a * (b*v - c*w)
    
    while t < t1:
        # Last step handling
        if t1 - t < 1e-14:
            break

        if t + h > t1:
            h = t1 - t
        
        # Stage 2
        v2 = v + h * (a21 * k1_v)
        w2 = w + h * (a21 * k1_w)
        k2_v = v2 - v2*v2*v2*one_third - w2 + I
        k2_w = a * (b*v2 - c*w2)
        
        # Stage 3
        v3 = v + h * (a31 * k1_v + a32 * k2_v)
        w3 = w + h * (a31 * k1_w + a32 * k2_w)
        k3_v = v3 - v3*v3*v3*one_third - w3 + I
        k3_w = a * (b*v3 - c*w3)
        
        # Stage 4
        v4 = v + h * (a41 * k1_v + a42 * k2_v + a43 * k3_v)
        w4 = w + h * (a41 * k1_w + a42 * k2_w + a43 * k3_w)
        k4_v = v4 - v4*v4*v4*one_third - w4 + I
        k4_w = a * (b*v4 - c*w4)
        
        # Stage 5
        v5 = v + h * (a51 * k1_v + a52 * k2_v + a53 * k3_v + a54 * k4_v)
        w5 = w + h * (a51 * k1_w + a52 * k2_w + a53 * k3_w + a54 * k4_w)
        k5_v = v5 - v5*v5*v5*one_third - w5 + I
        k5_w = a * (b*v5 - c*w5)
        
        # Stage 6
        v6 = v + h * (a61 * k1_v + a62 * k2_v + a63 * k3_v + a64 * k4_v + a65 * k5_v)
        w6 = w + h * (a61 * k1_w + a62 * k2_w + a63 * k3_w + a64 * k4_w + a65 * k5_w)
        k6_v = v6 - v6*v6*v6*one_third - w6 + I
        k6_w = a * (b*v6 - c*w6)
        
        # Candidate solution
        v_next = v + h * (b1 * k1_v + b3 * k3_v + b4 * k4_v + b5 * k5_v + b6 * k6_v)
        w_next = w + h * (b1 * k1_w + b3 * k3_w + b4 * k4_w + b5 * k5_w + b6 * k6_w)
        
        k7_v = v_next - v_next*v_next*v_next*one_third - w_next + I
        k7_w = a * (b*v_next - c*w_next)
        
        # Error estimate
        err_v = h * (e1 * k1_v + e3 * k3_v + e4 * k4_v + e5 * k5_v + e6 * k6_v + e7 * k7_v)
        err_w = h * (e1 * k1_w + e3 * k3_w + e4 * k4_w + e5 * k5_w + e6 * k6_w + e7 * k7_w)
        
        # Scale
        av = abs(v); avn = abs(v_next)
        scale_v = atol + rtol * (av if av > avn else avn)
        aw = abs(w); awn = abs(w_next)
        scale_w = atol + rtol * (aw if aw > awn else awn)
        
        # Error squared
        ratio_v = err_v / scale_v
        ratio_w = err_w / scale_w
        error_sq = 0.5 * (ratio_v*ratio_v + ratio_w*ratio_w)
        
        if error_sq < 1.0:
            t += h
            v = v_next
            w = w_next
            k1_v = k7_v
            k1_w = k7_w
            
            if error_sq == 0.0:
                h *= 10.0
            else:
                h *= min(10.0, 0.9 * error_sq**(-0.1))
        else:
            h *= max(0.1, 0.9 * error_sq**(-0.1))
            
    return v, w

class Solver:
    def __init__(self):
        # Trigger compilation
        solve_jit_scalar_opt(0.0, 0.0, 0.0, 1.0, 0.1, 0.1, 0.1, 0.1)

    def solve(self, problem, **kwargs) -> Any:
        y0 = problem["y0"]
        t0 = problem["t0"]
        t1 = problem["t1"]
        params = problem["params"]
        a = params["a"]
        b = params["b"]
        c = params["c"]
        I = params["I"]
        
        v, w = solve_jit_scalar_opt(y0[0], y0[1], t0, t1, a, b, c, I)
        return [v, w]