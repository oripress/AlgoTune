import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def brusselator_rhs(t, y, A, B):
    X, Y = y[0], y[1]
    dX = A + X * X * Y - (B + 1) * X
    dY = B * X - X * X * Y
    return dX, dY

@njit(cache=True, fastmath=True)
def dopri54_step(t, y, h, A, B):
    """Dormand-Prince 5(4) method - efficient embedded RK."""
    # Butcher tableau coefficients for DOPRI54
    c2, c3, c4, c5 = 0.2, 0.3, 0.8, 8.0/9.0
    
    a21 = 0.2
    a31, a32 = 3.0/40.0, 9.0/40.0
    a41, a42, a43 = 44.0/45.0, -56.0/15.0, 32.0/9.0
    a51, a52, a53, a54 = 19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0
    a61, a62, a63, a64, a65 = 9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0
    
    # 5th order solution coefficients
    b1, b3, b4, b5, b6 = 35.0/384.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0
    
    # 4th order solution coefficients (for error estimate)
    bs1, bs3, bs4, bs5, bs6, bs7 = 5179.0/57600.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0
    
    # Stage 1
    k1_x, k1_y = brusselator_rhs(t, y, A, B)
    
    # Stage 2
    y_temp0 = y[0] + h * a21 * k1_x
    y_temp1 = y[1] + h * a21 * k1_y
    k2_x, k2_y = brusselator_rhs(t + c2 * h, np.array([y_temp0, y_temp1]), A, B)
    
    # Stage 3
    y_temp0 = y[0] + h * (a31 * k1_x + a32 * k2_x)
    y_temp1 = y[1] + h * (a31 * k1_y + a32 * k2_y)
    k3_x, k3_y = brusselator_rhs(t + c3 * h, np.array([y_temp0, y_temp1]), A, B)
    
    # Stage 4
    y_temp0 = y[0] + h * (a41 * k1_x + a42 * k2_x + a43 * k3_x)
    y_temp1 = y[1] + h * (a41 * k1_y + a42 * k2_y + a43 * k3_y)
    k4_x, k4_y = brusselator_rhs(t + c4 * h, np.array([y_temp0, y_temp1]), A, B)
    
    # Stage 5
    y_temp0 = y[0] + h * (a51 * k1_x + a52 * k2_x + a53 * k3_x + a54 * k4_x)
    y_temp1 = y[1] + h * (a51 * k1_y + a52 * k2_y + a53 * k3_y + a54 * k4_y)
    k5_x, k5_y = brusselator_rhs(t + c5 * h, np.array([y_temp0, y_temp1]), A, B)
    
    # Stage 6
    y_temp0 = y[0] + h * (a61 * k1_x + a62 * k2_x + a63 * k3_x + a64 * k4_x + a65 * k5_x)
    y_temp1 = y[1] + h * (a61 * k1_y + a62 * k2_y + a63 * k3_y + a64 * k4_y + a65 * k5_y)
    k6_x, k6_y = brusselator_rhs(t + h, np.array([y_temp0, y_temp1]), A, B)
    
    # 5th order solution
    y5_0 = y[0] + h * (b1 * k1_x + b3 * k3_x + b4 * k4_x + b5 * k5_x + b6 * k6_x)
    y5_1 = y[1] + h * (b1 * k1_y + b3 * k3_y + b4 * k4_y + b5 * k5_y + b6 * k6_y)
    
    # Stage 7 (for 4th order solution)
    k7_x, k7_y = brusselator_rhs(t + h, np.array([y5_0, y5_1]), A, B)
    
    # 4th order solution (for error estimate)
    y4_0 = y[0] + h * (bs1 * k1_x + bs3 * k3_x + bs4 * k4_x + bs5 * k5_x + bs6 * k6_x + bs7 * k7_x)
    y4_1 = y[1] + h * (bs1 * k1_y + bs3 * k3_y + bs4 * k4_y + bs5 * k5_y + bs6 * k6_y + bs7 * k7_y)
    
    # Error estimate
    err0 = abs(y5_0 - y4_0)
    err1 = abs(y5_1 - y4_1)
    
    return np.array([y5_0, y5_1]), np.array([err0, err1])

@njit(cache=True, fastmath=True)
def adaptive_dopri54(y0, t0, t1, A, B, rtol=1e-8, atol=1e-8):
    """Adaptive DOPRI54 solver with PI step size control."""
    t = t0
    y = y0.copy()
    h = (t1 - t0) / 50.0  # Initial step size
    
    max_steps = 100000
    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0
    
    for _ in range(max_steps):
        if t >= t1:
            break
            
        # Don't overshoot
        if t + h > t1:
            h = t1 - t
        
        # Take step
        y_new, err = dopri54_step(t, y, h, A, B)
        
        # Compute relative error
        scale0 = atol + rtol * max(abs(y[0]), abs(y_new[0]))
        scale1 = atol + rtol * max(abs(y[1]), abs(y_new[1]))
        
        err_norm = max(err[0] / scale0, err[1] / scale1)
        
        if err_norm <= 1.0:
            # Accept step
            y = y_new
            t += h
            
            # PI step size controller
            if err_norm > 0:
                factor = safety * (1.0 / err_norm) ** 0.2
                factor = min(max_factor, max(min_factor, factor))
                h *= factor
        else:
            # Reject step, reduce step size
            factor = max(min_factor, safety * (1.0 / err_norm) ** 0.25)
            h *= factor
    
    return y

class Solver:
    def solve(self, problem, **kwargs):
        """Solve the Brusselator reaction model."""
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        A = problem["params"]["A"]
        B = problem["params"]["B"]
        
        result = adaptive_dopri54(y0, t0, t1, A, B, rtol=1e-8, atol=1e-8)
        
        return result