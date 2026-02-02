import numpy as np
from numba import njit
from scipy.integrate import ode

# Pre-compile the ODE function at module level
@njit(cache=True, fastmath=True)
def _fhn_rhs(v, w, a, b, c, I):
    dv = v - (v * v * v) / 3.0 - w + I
    dw = a * (b * v - c * w)
    return dv, dw

@njit(cache=True, fastmath=True)
def solve_rk45_fhn(y0_v, y0_w, t0, t1, a, b, c, I, rtol, atol):
    """Fully numba-compiled Dormand-Prince RK45 solver"""
    v = y0_v
    w = y0_w
    t = t0
    
    # Dormand-Prince coefficients
    c2, c3, c4, c5 = 0.2, 0.3, 0.8, 8.0/9.0
    
    a21 = 0.2
    a31, a32 = 0.075, 0.225
    a41, a42, a43 = 0.9777777777777777, -3.7333333333333334, 3.5555555555555554
    a51, a52, a53, a54 = 2.9525986892242035, -11.595793324188385, 9.822892851699436, -0.2908093278463649
    a61, a62, a63, a64, a65 = 2.8462752525252526, -10.757575757575758, 8.906422717743473, 0.2784090909090909, -0.2735313036020583
    
    # 5th order weights (b)
    b1, b3, b4, b5, b6 = 0.09114583333333333, 0.44923629829290207, 0.6510416666666666, -0.322376179245283, 0.13095238095238096
    
    # Error coefficients (b - b*)
    e1, e3, e4, e5, e6, e7 = 0.0012326388888888888, -0.0042527702905061394, 0.03697916666666667, -0.05086379716981132, 0.0419047619047619, -0.025
    
    # Initial step size
    f0_v, f0_w = _fhn_rhs(v, w, a, b, c, I)
    d0 = max(abs(v), abs(w)) + 1e-10
    d1 = max(abs(f0_v), abs(f0_w)) + 1e-10
    h0 = 0.01 * d0 / d1
    
    v1 = v + h0 * f0_v
    w1 = w + h0 * f0_w
    f1_v, f1_w = _fhn_rhs(v1, w1, a, b, c, I)
    d2 = max(abs(f1_v - f0_v), abs(f1_w - f0_w)) / h0 + 1e-10
    
    if max(d1, d2) <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** 0.2
    
    h = min(100 * h0, h1, t1 - t0)
    
    # Pre-allocate
    k1_v, k1_w = f0_v, f0_w
    
    while t < t1:
        if t + h > t1:
            h = t1 - t
        
        # RK45 stages
        k2_v, k2_w = _fhn_rhs(v + h*a21*k1_v, w + h*a21*k1_w, a, b, c, I)
        k3_v, k3_w = _fhn_rhs(v + h*(a31*k1_v + a32*k2_v), w + h*(a31*k1_w + a32*k2_w), a, b, c, I)
        k4_v, k4_w = _fhn_rhs(v + h*(a41*k1_v + a42*k2_v + a43*k3_v), w + h*(a41*k1_w + a42*k2_w + a43*k3_w), a, b, c, I)
        k5_v, k5_w = _fhn_rhs(v + h*(a51*k1_v + a52*k2_v + a53*k3_v + a54*k4_v), w + h*(a51*k1_w + a52*k2_w + a53*k3_w + a54*k4_w), a, b, c, I)
        k6_v, k6_w = _fhn_rhs(v + h*(a61*k1_v + a62*k2_v + a63*k3_v + a64*k4_v + a65*k5_v), w + h*(a61*k1_w + a62*k2_w + a63*k3_w + a64*k4_w + a65*k5_w), a, b, c, I)
        
        # 5th order solution
        v_new = v + h * (b1*k1_v + b3*k3_v + b4*k4_v + b5*k5_v + b6*k6_v)
        w_new = w + h * (b1*k1_w + b3*k3_w + b4*k4_w + b5*k5_w + b6*k6_w)
        
        k7_v, k7_w = _fhn_rhs(v_new, w_new, a, b, c, I)
        
        # Error estimate
        err_v = h * (e1*k1_v + e3*k3_v + e4*k4_v + e5*k5_v + e6*k6_v + e7*k7_v)
        err_w = h * (e1*k1_w + e3*k3_w + e4*k4_w + e5*k5_w + e6*k6_w + e7*k7_w)
        
        sc_v = atol + rtol * max(abs(v), abs(v_new))
        sc_w = atol + rtol * max(abs(w), abs(w_new))
        
        err_norm = np.sqrt(0.5 * ((err_v/sc_v)**2 + (err_w/sc_w)**2))
        
        if err_norm <= 1.0:
            t = t + h
            v = v_new
            w = w_new
            k1_v, k1_w = k7_v, k7_w  # FSAL
        
        # Step size adjustment
        if err_norm > 1e-10:
            factor = 0.9 * err_norm ** (-0.2)
            if err_norm > 1.0:
                factor = max(0.2, factor)
            else:
                factor = min(10.0, factor)
            h = h * factor
        else:
            h = h * 10.0
        
        h = min(h, t1 - t0)
        h = max(h, 1e-14)
    
    return v, w

class Solver:
    def solve(self, problem, **kwargs):
        y0 = problem["y0"]
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        
        a = float(params["a"])
        b = float(params["b"])
        c = float(params["c"])
        I = float(params["I"])
        
        v, w = solve_rk45_fhn(float(y0[0]), float(y0[1]), t0, t1, a, b, c, I, 1e-8, 1e-8)
        
        return [v, w]