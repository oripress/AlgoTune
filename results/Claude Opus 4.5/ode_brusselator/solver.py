from typing import Any
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def solve_brusselator(y0_0, y0_1, t0, t1, A, B, rtol, atol):
    """Solve Brusselator ODE using adaptive RK45 with FSAL optimization."""
    B_plus_1 = B + 1.0
    
    y0 = y0_0
    y1 = y0_1
    t = t0
    h = (t1 - t0) / 100.0
    
    safety = 0.9
    min_factor = 0.2
    max_factor = 5.0
    
    # Initial k1
    X2Y = y0 * y0 * y1
    k1_0 = A + X2Y - B_plus_1 * y0
    k1_1 = B * y0 - X2Y
    
    while t < t1:
        if t + h > t1:
            h = t1 - t
        
        # k2 (k1 already computed or from FSAL)
        yt0 = y0 + h * 0.2 * k1_0
        yt1 = y1 + h * 0.2 * k1_1
        X2Y = yt0 * yt0 * yt1
        k2_0 = A + X2Y - B_plus_1 * yt0
        k2_1 = B * yt0 - X2Y
        
        # k3
        yt0 = y0 + h * (0.075 * k1_0 + 0.225 * k2_0)
        yt1 = y1 + h * (0.075 * k1_1 + 0.225 * k2_1)
        X2Y = yt0 * yt0 * yt1
        k3_0 = A + X2Y - B_plus_1 * yt0
        k3_1 = B * yt0 - X2Y
        
        # k4
        yt0 = y0 + h * (0.9777777777777777 * k1_0 - 3.7333333333333334 * k2_0 + 3.5555555555555554 * k3_0)
        yt1 = y1 + h * (0.9777777777777777 * k1_1 - 3.7333333333333334 * k2_1 + 3.5555555555555554 * k3_1)
        X2Y = yt0 * yt0 * yt1
        k4_0 = A + X2Y - B_plus_1 * yt0
        k4_1 = B * yt0 - X2Y
        
        # k5
        yt0 = y0 + h * (2.9525986892242035 * k1_0 - 11.595793324188385 * k2_0 + 9.822892851699436 * k3_0 - 0.2908093278463649 * k4_0)
        yt1 = y1 + h * (2.9525986892242035 * k1_1 - 11.595793324188385 * k2_1 + 9.822892851699436 * k3_1 - 0.2908093278463649 * k4_1)
        X2Y = yt0 * yt0 * yt1
        k5_0 = A + X2Y - B_plus_1 * yt0
        k5_1 = B * yt0 - X2Y
        
        # k6
        yt0 = y0 + h * (2.8462752525252526 * k1_0 - 10.757575757575758 * k2_0 + 8.906422717743473 * k3_0 + 0.2784090909090909 * k4_0 - 0.2735313036020583 * k5_0)
        yt1 = y1 + h * (2.8462752525252526 * k1_1 - 10.757575757575758 * k2_1 + 8.906422717743473 * k3_1 + 0.2784090909090909 * k4_1 - 0.2735313036020583 * k5_1)
        X2Y = yt0 * yt0 * yt1
        k6_0 = A + X2Y - B_plus_1 * yt0
        k6_1 = B * yt0 - X2Y
        
        # 5th order solution
        yn0 = y0 + h * (0.09114583333333333 * k1_0 + 0.44923629829290207 * k3_0 + 0.6510416666666666 * k4_0 - 0.322376179245283 * k5_0 + 0.13095238095238096 * k6_0)
        yn1 = y1 + h * (0.09114583333333333 * k1_1 + 0.44923629829290207 * k3_1 + 0.6510416666666666 * k4_1 - 0.322376179245283 * k5_1 + 0.13095238095238096 * k6_1)
        
        # k7 (also k1 for next step via FSAL)
        X2Y = yn0 * yn0 * yn1
        k7_0 = A + X2Y - B_plus_1 * yn0
        k7_1 = B * yn0 - X2Y
        
        # Error
        err0 = h * abs(0.0012326388888888888 * k1_0 - 0.0042527702905061394 * k3_0 + 0.03697916666666667 * k4_0 - 0.05086379716981132 * k5_0 + 0.041904761904761906 * k6_0 - 0.025 * k7_0)
        err1 = h * abs(0.0012326388888888888 * k1_1 - 0.0042527702905061394 * k3_1 + 0.03697916666666667 * k4_1 - 0.05086379716981132 * k5_1 + 0.041904761904761906 * k6_1 - 0.025 * k7_1)
        
        scale0 = atol + rtol * max(abs(y0), abs(yn0))
        scale1 = atol + rtol * max(abs(y1), abs(yn1))
        err_norm = ((err0 / scale0) ** 2 + (err1 / scale1) ** 2) ** 0.5 * 0.7071067811865476
        
        if err_norm <= 1.0:
            t = t + h
            y0, y1 = yn0, yn1
            # FSAL: k7 becomes k1 for next step
            k1_0, k1_1 = k7_0, k7_1
            
            if err_norm < 1e-10:
                factor = max_factor
            else:
                factor = safety * err_norm ** (-0.2)
                if factor < min_factor:
                    factor = min_factor
                elif factor > max_factor:
                    factor = max_factor
            h = h * factor
        else:
            factor = safety * err_norm ** (-0.25)
            if factor < min_factor:
                factor = min_factor
            h = h * factor
            # Recompute k1 since step was rejected and y didn't change
            X2Y = y0 * y0 * y1
            k1_0 = A + X2Y - B_plus_1 * y0
            k1_1 = B * y0 - X2Y
    
    return y0, y1

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        y0 = problem["y0"]
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        params = problem["params"]
        
        A = float(params["A"])
        B = float(params["B"])
        
        r0, r1 = solve_brusselator(float(y0[0]), float(y0[1]), t0, t1, A, B, 1e-8, 1e-8)
        
        return np.array([r0, r1])