from typing import Any
import numpy as np
from numba import njit
import math

@njit(cache=True, fastmath=True)
def hodgkin_huxley_rhs(V, m, h, n, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app):
    # Rate constants
    x = V + 40.0
    if abs(x) < 1e-7:
        alpha_m = 1.0
    else:
        alpha_m = 0.1 * x / (1.0 - math.exp(-x * 0.1))
    
    beta_m = 4.0 * math.exp(-(V + 65.0) / 18.0)
    alpha_h = 0.07 * math.exp(-(V + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))
    
    x = V + 55.0
    if abs(x) < 1e-7:
        alpha_n = 0.1
    else:
        alpha_n = 0.01 * x / (1.0 - math.exp(-x * 0.1))
    
    beta_n = 0.125 * math.exp(-(V + 65.0) / 80.0)
    
    # Currents
    m3 = m * m * m
    n4 = n * n * n * n
    I_Na = g_Na * m3 * h * (V - E_Na)
    I_K = g_K * n4 * (V - E_K)
    I_L = g_L * (V - E_L)
    
    dVdt = (I_app - I_Na - I_K - I_L) / C_m
    dmdt = alpha_m * (1.0 - m) - beta_m * m
    dhdt = alpha_h * (1.0 - h) - beta_h * h
    dndt = alpha_n * (1.0 - n) - beta_n * n
    
    return dVdt, dmdt, dhdt, dndt

@njit(cache=True, fastmath=True)
def rk45_step(V, m, h, n, t, h_step, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app):
    # RK45 Dormand-Prince coefficients
    c2, c3, c4, c5, c6 = 1/5, 3/10, 4/5, 8/9, 1.0
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    
    b1, b2, b3, b4, b5, b6 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
    e1, e2, e3, e4, e5, e6, e7 = 71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40
    
    # k1
    k1V, k1m, k1h, k1n = hodgkin_huxley_rhs(V, m, h, n, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    # k2
    V2 = V + h_step * a21 * k1V
    m2 = m + h_step * a21 * k1m
    h2 = h + h_step * a21 * k1h
    n2 = n + h_step * a21 * k1n
    k2V, k2m, k2h, k2n = hodgkin_huxley_rhs(V2, m2, h2, n2, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    # k3
    V3 = V + h_step * (a31 * k1V + a32 * k2V)
    m3 = m + h_step * (a31 * k1m + a32 * k2m)
    h3 = h + h_step * (a31 * k1h + a32 * k2h)
    n3 = n + h_step * (a31 * k1n + a32 * k2n)
    k3V, k3m, k3h, k3n = hodgkin_huxley_rhs(V3, m3, h3, n3, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    # k4
    V4 = V + h_step * (a41 * k1V + a42 * k2V + a43 * k3V)
    m4 = m + h_step * (a41 * k1m + a42 * k2m + a43 * k3m)
    h4 = h + h_step * (a41 * k1h + a42 * k2h + a43 * k3h)
    n4 = n + h_step * (a41 * k1n + a42 * k2n + a43 * k3n)
    k4V, k4m, k4h, k4n = hodgkin_huxley_rhs(V4, m4, h4, n4, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    # k5
    V5 = V + h_step * (a51 * k1V + a52 * k2V + a53 * k3V + a54 * k4V)
    m5 = m + h_step * (a51 * k1m + a52 * k2m + a53 * k3m + a54 * k4m)
    h5 = h + h_step * (a51 * k1h + a52 * k2h + a53 * k3h + a54 * k4h)
    n5 = n + h_step * (a51 * k1n + a52 * k2n + a53 * k3n + a54 * k4n)
    k5V, k5m, k5h, k5n = hodgkin_huxley_rhs(V5, m5, h5, n5, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    # k6
    V6 = V + h_step * (a61 * k1V + a62 * k2V + a63 * k3V + a64 * k4V + a65 * k5V)
    m6 = m + h_step * (a61 * k1m + a62 * k2m + a63 * k3m + a64 * k4m + a65 * k5m)
    h6 = h + h_step * (a61 * k1h + a62 * k2h + a63 * k3h + a64 * k4h + a65 * k5h)
    n6 = n + h_step * (a61 * k1n + a62 * k2n + a63 * k3n + a64 * k4n + a65 * k5n)
    k6V, k6m, k6h, k6n = hodgkin_huxley_rhs(V6, m6, h6, n6, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    # 5th order solution
    Vnew = V + h_step * (b1 * k1V + b3 * k3V + b4 * k4V + b5 * k5V + b6 * k6V)
    mnew = m + h_step * (b1 * k1m + b3 * k3m + b4 * k4m + b5 * k5m + b6 * k6m)
    hnew = h + h_step * (b1 * k1h + b3 * k3h + b4 * k4h + b5 * k5h + b6 * k6h)
    nnew = n + h_step * (b1 * k1n + b3 * k3n + b4 * k4n + b5 * k5n + b6 * k6n)
    
    # k7 for error estimate
    k7V, k7m, k7h, k7n = hodgkin_huxley_rhs(Vnew, mnew, hnew, nnew, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)
    
    # Error estimate
    errV = h_step * (e1 * k1V + e3 * k3V + e4 * k4V + e5 * k5V + e6 * k6V + e7 * k7V)
    errm = h_step * (e1 * k1m + e3 * k3m + e4 * k4m + e5 * k5m + e6 * k6m + e7 * k7m)
    errh = h_step * (e1 * k1h + e3 * k3h + e4 * k4h + e5 * k5h + e6 * k6h + e7 * k7h)
    errn = h_step * (e1 * k1n + e3 * k3n + e4 * k4n + e5 * k5n + e6 * k6n + e7 * k7n)
    
    return Vnew, mnew, hnew, nnew, errV, errm, errh, errn

@njit(cache=True, fastmath=True)
def solve_hh(t0, t1, V0, m0, h0, n0, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app, rtol, atol):
    t = t0
    V, m, h, n = V0, m0, h0, n0
    
    # Initial step size
    h_step = (t1 - t0) / 1000.0
    h_min = 1e-12
    h_max = (t1 - t0) / 10.0
    
    max_iter = 1000000
    for _ in range(max_iter):
        if t >= t1:
            break
        
        # Don't overshoot
        if t + h_step > t1:
            h_step = t1 - t
        
        Vnew, mnew, hnew, nnew, errV, errm, errh, errn = rk45_step(
            V, m, h, n, t, h_step, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app
        )
        
        # Calculate error
        sc_V = atol + rtol * max(abs(V), abs(Vnew))
        sc_m = atol + rtol * max(abs(m), abs(mnew))
        sc_h = atol + rtol * max(abs(h), abs(hnew))
        sc_n = atol + rtol * max(abs(n), abs(nnew))
        
        err = math.sqrt(0.25 * ((errV/sc_V)**2 + (errm/sc_m)**2 + (errh/sc_h)**2 + (errn/sc_n)**2))
        
        if err <= 1.0:
            # Accept step
            t = t + h_step
            V, m, h, n = Vnew, mnew, hnew, nnew
        
        # Adjust step size
        if err > 0:
            factor = 0.9 * err**(-0.2)
            factor = min(5.0, max(0.2, factor))
            h_step = h_step * factor
        else:
            h_step = h_step * 5.0
        
        h_step = max(h_min, min(h_max, h_step))
    
    return V, m, h, n

# Warm up compilation
_dummy = solve_hh(0.0, 0.1, -65.0, 0.05, 0.6, 0.3, 1.0, 120.0, 36.0, 0.3, 50.0, -77.0, -54.4, 10.0, 1e-8, 1e-8)

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        y0 = problem["y0"]
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        V, m, h, n = solve_hh(
            t0, t1, y0[0], y0[1], y0[2], y0[3],
            params["C_m"], params["g_Na"], params["g_K"], params["g_L"],
            params["E_Na"], params["E_K"], params["E_L"], params["I_app"],
            1e-8, 1e-8
        )
        
        return [V, m, h, n]