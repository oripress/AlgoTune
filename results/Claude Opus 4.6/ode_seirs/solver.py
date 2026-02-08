import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def rk45_solve(t0, t1, y0, beta, sigma, gamma, omega, rtol, atol):
    """Adaptive RK45 (Dormand-Prince) solver implemented in numba."""
    # Dormand-Prince coefficients
    b21 = 1.0/5.0
    b31 = 3.0/40.0
    b32 = 9.0/40.0
    b41 = 44.0/45.0
    b42 = -56.0/15.0
    b43 = 32.0/9.0
    b51 = 19372.0/6561.0
    b52 = -25360.0/2187.0
    b53 = 64448.0/6561.0
    b54 = -212.0/729.0
    b61 = 9017.0/3168.0
    b62 = -355.0/33.0
    b63 = 46732.0/5247.0
    b64 = 49.0/176.0
    b65 = -5103.0/18656.0
    
    # 5th order weights
    c1 = 35.0/384.0
    c3 = 500.0/1113.0
    c4 = 125.0/192.0
    c5 = -2187.0/6784.0
    c6 = 11.0/84.0
    
    # Error coefficients
    e1 = 71.0/57600.0
    e3 = -71.0/16695.0
    e4 = 71.0/1920.0
    e5 = -17253.0/339200.0
    e6 = 22.0/525.0
    e7 = -1.0/40.0
    
    t = t0
    S = y0[0]
    E = y0[1]
    I = y0[2]
    R = y0[3]
    
    # Initial step size estimate
    dt = t1 - t0
    # Compute initial derivatives for step size estimate
    bSI = beta * S * I
    f0 = max(abs(-bSI + omega * R), abs(bSI - sigma * E), 
             abs(sigma * E - gamma * I), abs(gamma * I - omega * R))
    if f0 > 1e-10:
        h = min(dt * 0.1, 0.01 / f0)
    else:
        h = dt * 0.1
    h = min(h, dt)
    
    while t < t1:
        if t + h > t1:
            h = t1 - t
        if h < 1e-14:
            break
        
        # k1
        bSI = beta * S * I
        k1_0 = -bSI + omega * R
        k1_1 = bSI - sigma * E
        k1_2 = sigma * E - gamma * I
        k1_3 = gamma * I - omega * R
        
        # k2
        S2 = S + h * b21 * k1_0
        E2 = E + h * b21 * k1_1
        I2 = I + h * b21 * k1_2
        R2 = R + h * b21 * k1_3
        bSI = beta * S2 * I2
        k2_0 = -bSI + omega * R2
        k2_1 = bSI - sigma * E2
        k2_2 = sigma * E2 - gamma * I2
        k2_3 = gamma * I2 - omega * R2
        
        # k3
        S3 = S + h * (b31 * k1_0 + b32 * k2_0)
        E3 = E + h * (b31 * k1_1 + b32 * k2_1)
        I3 = I + h * (b31 * k1_2 + b32 * k2_2)
        R3 = R + h * (b31 * k1_3 + b32 * k2_3)
        bSI = beta * S3 * I3
        k3_0 = -bSI + omega * R3
        k3_1 = bSI - sigma * E3
        k3_2 = sigma * E3 - gamma * I3
        k3_3 = gamma * I3 - omega * R3
        
        # k4
        S4 = S + h * (b41 * k1_0 + b42 * k2_0 + b43 * k3_0)
        E4 = E + h * (b41 * k1_1 + b42 * k2_1 + b43 * k3_1)
        I4 = I + h * (b41 * k1_2 + b42 * k2_2 + b43 * k3_2)
        R4 = R + h * (b41 * k1_3 + b42 * k2_3 + b43 * k3_3)
        bSI = beta * S4 * I4
        k4_0 = -bSI + omega * R4
        k4_1 = bSI - sigma * E4
        k4_2 = sigma * E4 - gamma * I4
        k4_3 = gamma * I4 - omega * R4
        
        # k5
        S5 = S + h * (b51 * k1_0 + b52 * k2_0 + b53 * k3_0 + b54 * k4_0)
        E5 = E + h * (b51 * k1_1 + b52 * k2_1 + b53 * k3_1 + b54 * k4_1)
        I5 = I + h * (b51 * k1_2 + b52 * k2_2 + b53 * k3_2 + b54 * k4_2)
        R5 = R + h * (b51 * k1_3 + b52 * k2_3 + b53 * k3_3 + b54 * k4_3)
        bSI = beta * S5 * I5
        k5_0 = -bSI + omega * R5
        k5_1 = bSI - sigma * E5
        k5_2 = sigma * E5 - gamma * I5
        k5_3 = gamma * I5 - omega * R5
        
        # k6
        S6 = S + h * (b61 * k1_0 + b62 * k2_0 + b63 * k3_0 + b64 * k4_0 + b65 * k5_0)
        E6 = E + h * (b61 * k1_1 + b62 * k2_1 + b63 * k3_1 + b64 * k4_1 + b65 * k5_1)
        I6 = I + h * (b61 * k1_2 + b62 * k2_2 + b63 * k3_2 + b64 * k4_2 + b65 * k5_2)
        R6 = R + h * (b61 * k1_3 + b62 * k2_3 + b63 * k3_3 + b64 * k4_3 + b65 * k5_3)
        bSI = beta * S6 * I6
        k6_0 = -bSI + omega * R6
        k6_1 = bSI - sigma * E6
        k6_2 = sigma * E6 - gamma * I6
        k6_3 = gamma * I6 - omega * R6
        
        # 5th order solution
        Sn = S + h * (c1 * k1_0 + c3 * k3_0 + c4 * k4_0 + c5 * k5_0 + c6 * k6_0)
        En = E + h * (c1 * k1_1 + c3 * k3_1 + c4 * k4_1 + c5 * k5_1 + c6 * k6_1)
        In = I + h * (c1 * k1_2 + c3 * k3_2 + c4 * k4_2 + c5 * k5_2 + c6 * k6_2)
        Rn = R + h * (c1 * k1_3 + c3 * k3_3 + c4 * k4_3 + c5 * k5_3 + c6 * k6_3)
        
        # k7 for error estimation
        bSI = beta * Sn * In
        k7_0 = -bSI + omega * Rn
        k7_1 = bSI - sigma * En
        k7_2 = sigma * En - gamma * In
        k7_3 = gamma * In - omega * Rn
        
        # Error estimation
        err_0 = h * (e1 * k1_0 + e3 * k3_0 + e4 * k4_0 + e5 * k5_0 + e6 * k6_0 + e7 * k7_0)
        err_1 = h * (e1 * k1_1 + e3 * k3_1 + e4 * k4_1 + e5 * k5_1 + e6 * k6_1 + e7 * k7_1)
        err_2 = h * (e1 * k1_2 + e3 * k3_2 + e4 * k4_2 + e5 * k5_2 + e6 * k6_2 + e7 * k7_2)
        err_3 = h * (e1 * k1_3 + e3 * k3_3 + e4 * k4_3 + e5 * k5_3 + e6 * k6_3 + e7 * k7_3)
        
        # Compute error norm
        sc_0 = atol + rtol * max(abs(S), abs(Sn))
        sc_1 = atol + rtol * max(abs(E), abs(En))
        sc_2 = atol + rtol * max(abs(I), abs(In))
        sc_3 = atol + rtol * max(abs(R), abs(Rn))
        
        err_norm = ((err_0/sc_0)**2 + (err_1/sc_1)**2 + (err_2/sc_2)**2 + (err_3/sc_3)**2)**0.5 * 0.5
        
        if err_norm <= 1.0:
            t += h
            S = Sn
            E = En
            I = In
            R = Rn
        
        # Step size adjustment
        if err_norm < 1e-15:
            h *= 5.0
        else:
            h *= min(5.0, max(0.2, 0.9 * err_norm**(-0.2)))
    
    y = np.empty(4)
    y[0] = S
    y[1] = E
    y[2] = I
    y[3] = R
    return y

class Solver:
    def __init__(self):
        # Warm up numba compilation
        y0 = np.array([0.89, 0.01, 0.005, 0.095])
        rk45_solve(0.0, 1.0, y0, 0.35, 0.2, 0.1, 0.002, 1e-8, 1e-8)
    
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        
        beta = float(params["beta"])
        sigma = float(params["sigma"])
        gamma = float(params["gamma"])
        omega = float(params["omega"])
        
        result = rk45_solve(t0, t1, y0, beta, sigma, gamma, omega, 1e-8, 1e-8)
        return result.tolist()