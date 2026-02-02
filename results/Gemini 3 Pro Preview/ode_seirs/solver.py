import numpy as np
from numba import njit
import math

@njit(fastmath=True)
def solve_seirs(y0, t0, t1, beta, sigma, gamma, omega, rtol, atol):
    S, E, I, R = y0[0], y0[1], y0[2], y0[3]
    t = t0
    
    # Initial step size guess
    h = 1e-4 * (t1 - t0)
    if h == 0: h = 1e-6
    
    # Initial derivative
    k1_S = -beta * S * I + omega * R
    k1_E = beta * S * I - sigma * E
    k1_I = sigma * E - gamma * I
    k1_R = gamma * I - omega * R
    
    max_steps = 10000000
    step = 0
    
    while t < t1 and step < max_steps:
        step += 1
        if t + h > t1:
            h = t1 - t
            
        # k2
        S2 = S + h * (0.2 * k1_S)
        E2 = E + h * (0.2 * k1_E)
        I2 = I + h * (0.2 * k1_I)
        R2 = R + h * (0.2 * k1_R)
        
        k2_S = -beta * S2 * I2 + omega * R2
        k2_E = beta * S2 * I2 - sigma * E2
        k2_I = sigma * E2 - gamma * I2
        k2_R = gamma * I2 - omega * R2
        
        # k3
        S3 = S + h * (0.075*k1_S + 0.225*k2_S)
        E3 = E + h * (0.075*k1_E + 0.225*k2_E)
        I3 = I + h * (0.075*k1_I + 0.225*k2_I)
        R3 = R + h * (0.075*k1_R + 0.225*k2_R)
        
        k3_S = -beta * S3 * I3 + omega * R3
        k3_E = beta * S3 * I3 - sigma * E3
        k3_I = sigma * E3 - gamma * I3
        k3_R = gamma * I3 - omega * R3
        
        # k4
        S4 = S + h * ((44/45)*k1_S + (-56/15)*k2_S + (32/9)*k3_S)
        E4 = E + h * ((44/45)*k1_E + (-56/15)*k2_E + (32/9)*k3_E)
        I4 = I + h * ((44/45)*k1_I + (-56/15)*k2_I + (32/9)*k3_I)
        R4 = R + h * ((44/45)*k1_R + (-56/15)*k2_R + (32/9)*k3_R)
        
        k4_S = -beta * S4 * I4 + omega * R4
        k4_E = beta * S4 * I4 - sigma * E4
        k4_I = sigma * E4 - gamma * I4
        k4_R = gamma * I4 - omega * R4
        
        # k5
        S5 = S + h * ((19372/6561)*k1_S + (-25360/2187)*k2_S + (64448/6561)*k3_S + (-212/729)*k4_S)
        E5 = E + h * ((19372/6561)*k1_E + (-25360/2187)*k2_E + (64448/6561)*k3_E + (-212/729)*k4_E)
        I5 = I + h * ((19372/6561)*k1_I + (-25360/2187)*k2_I + (64448/6561)*k3_I + (-212/729)*k4_I)
        R5 = R + h * ((19372/6561)*k1_R + (-25360/2187)*k2_R + (64448/6561)*k3_R + (-212/729)*k4_R)
        
        k5_S = -beta * S5 * I5 + omega * R5
        k5_E = beta * S5 * I5 - sigma * E5
        k5_I = sigma * E5 - gamma * I5
        k5_R = gamma * I5 - omega * R5
        
        # k6
        S6 = S + h * ((9017/3168)*k1_S + (-355/33)*k2_S + (46732/5247)*k3_S + (49/176)*k4_S + (-5103/18656)*k5_S)
        E6 = E + h * ((9017/3168)*k1_E + (-355/33)*k2_E + (46732/5247)*k3_E + (49/176)*k4_E + (-5103/18656)*k5_E)
        I6 = I + h * ((9017/3168)*k1_I + (-355/33)*k2_I + (46732/5247)*k3_I + (49/176)*k4_I + (-5103/18656)*k5_I)
        R6 = R + h * ((9017/3168)*k1_R + (-355/33)*k2_R + (46732/5247)*k3_R + (49/176)*k4_R + (-5103/18656)*k5_R)
        
        k6_S = -beta * S6 * I6 + omega * R6
        k6_E = beta * S6 * I6 - sigma * E6
        k6_I = sigma * E6 - gamma * I6
        k6_R = gamma * I6 - omega * R6
        
        # k7 (same as y_new)
        S_new = S + h * ((35/384)*k1_S + (500/1113)*k3_S + (125/192)*k4_S + (-2187/6784)*k5_S + (11/84)*k6_S)
        E_new = E + h * ((35/384)*k1_E + (500/1113)*k3_E + (125/192)*k4_E + (-2187/6784)*k5_E + (11/84)*k6_E)
        I_new = I + h * ((35/384)*k1_I + (500/1113)*k3_I + (125/192)*k4_I + (-2187/6784)*k5_I + (11/84)*k6_I)
        R_new = R + h * ((35/384)*k1_R + (500/1113)*k3_R + (125/192)*k4_R + (-2187/6784)*k5_R + (11/84)*k6_R)
        
        k7_S = -beta * S_new * I_new + omega * R_new
        k7_E = beta * S_new * I_new - sigma * E_new
        k7_I = sigma * E_new - gamma * I_new
        k7_R = gamma * I_new - omega * R_new
        
        # Error estimation
        err_S = h * ((-71/57600)*k1_S + (71/16695)*k3_S + (-71/1920)*k4_S + (17253/339200)*k5_S + (-22/525)*k6_S + (1/40)*k7_S)
        err_E = h * ((-71/57600)*k1_E + (71/16695)*k3_E + (-71/1920)*k4_E + (17253/339200)*k5_E + (-22/525)*k6_E + (1/40)*k7_E)
        err_I = h * ((-71/57600)*k1_I + (71/16695)*k3_I + (-71/1920)*k4_I + (17253/339200)*k5_I + (-22/525)*k6_I + (1/40)*k7_I)
        err_R = h * ((-71/57600)*k1_R + (71/16695)*k3_R + (-71/1920)*k4_R + (17253/339200)*k5_R + (-22/525)*k6_R + (1/40)*k7_R)
        
        # Error norm
        sc_S = atol + rtol * max(abs(S), abs(S_new))
        sc_E = atol + rtol * max(abs(E), abs(E_new))
        sc_I = atol + rtol * max(abs(I), abs(I_new))
        sc_R = atol + rtol * max(abs(R), abs(R_new))
        
        err_ratio_sq = (err_S/sc_S)**2 + (err_E/sc_E)**2 + (err_I/sc_I)**2 + (err_R/sc_R)**2
        error_norm = np.sqrt(err_ratio_sq / 4.0)
        
        if error_norm < 1.0:
            t += h
            S, E, I, R = S_new, E_new, I_new, R_new
            
            # FSAL: k7 becomes k1 for the next step
            k1_S, k1_E, k1_I, k1_R = k7_S, k7_E, k7_I, k7_R
            
            if error_norm == 0:
                factor = 10.0
            else:
                factor = 0.9 * error_norm**(-0.2)
        else:
            # Reject step, k1 remains the same
            if error_norm == 0:
                factor = 0.1
            else:
                factor = 0.9 * error_norm**(-0.2)
                
        if factor < 0.2: factor = 0.2
        if factor > 10.0: factor = 10.0
        h *= factor
        
        if t >= t1 or abs(t1 - t) < 1e-14:
            break
            
    return np.array([S, E, I, R])

class Solver:
    def solve(self, problem, **kwargs):
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        
        beta = float(params["beta"])
        sigma = float(params["sigma"])
        gamma = float(params["gamma"])
        omega = float(params["omega"])
        
        y_final = solve_seirs(y0, t0, t1, beta, sigma, gamma, omega, 1e-6, 1e-8)
        
        return y_final.tolist()