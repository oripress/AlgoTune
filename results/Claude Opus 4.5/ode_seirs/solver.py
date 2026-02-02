from typing import Any
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def integrate_seirs_scalar(S0, E0, I0, R0, t0, t1, beta, sigma, gamma, omega, rtol, atol):
    S, E, I, R = S0, E0, I0, R0
    t = t0
    h = min(1.0, (t1 - t0) / 10)
    
    while t < t1:
        if t + h > t1:
            h = t1 - t
        
        # k1
        bSI = beta * S * I
        k1S = -bSI + omega * R
        k1E = bSI - sigma * E
        k1I = sigma * E - gamma * I
        k1R = gamma * I - omega * R
        
        # k2
        tmpS = S + h * 0.25 * k1S
        tmpE = E + h * 0.25 * k1E
        tmpI = I + h * 0.25 * k1I
        tmpR = R + h * 0.25 * k1R
        bSI = beta * tmpS * tmpI
        k2S = -bSI + omega * tmpR
        k2E = bSI - sigma * tmpE
        k2I = sigma * tmpE - gamma * tmpI
        k2R = gamma * tmpI - omega * tmpR
        
        # k3
        tmpS = S + h * (0.09375 * k1S + 0.28125 * k2S)
        tmpE = E + h * (0.09375 * k1E + 0.28125 * k2E)
        tmpI = I + h * (0.09375 * k1I + 0.28125 * k2I)
        tmpR = R + h * (0.09375 * k1R + 0.28125 * k2R)
        bSI = beta * tmpS * tmpI
        k3S = -bSI + omega * tmpR
        k3E = bSI - sigma * tmpE
        k3I = sigma * tmpE - gamma * tmpI
        k3R = gamma * tmpI - omega * tmpR
        
        # k4
        c41 = 0.8793809740555303  # 1932/2197
        c42 = -3.2772556710137806  # -7200/2197
        c43 = 3.3208911449643374  # 7296/2197
        tmpS = S + h * (c41 * k1S + c42 * k2S + c43 * k3S)
        tmpE = E + h * (c41 * k1E + c42 * k2E + c43 * k3E)
        tmpI = I + h * (c41 * k1I + c42 * k2I + c43 * k3I)
        tmpR = R + h * (c41 * k1R + c42 * k2R + c43 * k3R)
        bSI = beta * tmpS * tmpI
        k4S = -bSI + omega * tmpR
        k4E = bSI - sigma * tmpE
        k4I = sigma * tmpE - gamma * tmpI
        k4R = gamma * tmpI - omega * tmpR
        
        # k5
        c51 = 2.0324074074074074  # 439/216
        c53 = 7.1734892787524366  # 3680/513
        c54 = -0.20589668615984405  # -845/4104
        tmpS = S + h * (c51 * k1S - 8.0 * k2S + c53 * k3S + c54 * k4S)
        tmpE = E + h * (c51 * k1E - 8.0 * k2E + c53 * k3E + c54 * k4E)
        tmpI = I + h * (c51 * k1I - 8.0 * k2I + c53 * k3I + c54 * k4I)
        tmpR = R + h * (c51 * k1R - 8.0 * k2R + c53 * k3R + c54 * k4R)
        bSI = beta * tmpS * tmpI
        k5S = -bSI + omega * tmpR
        k5E = bSI - sigma * tmpE
        k5I = sigma * tmpE - gamma * tmpI
        k5R = gamma * tmpI - omega * tmpR
        
        # k6
        c61 = -0.2962962962962963  # -8/27
        c63 = -1.3816764132553606  # -3544/2565
        c64 = 0.45297270955165696  # 1859/4104
        c65 = -0.275  # -11/40
        tmpS = S + h * (c61 * k1S + 2.0 * k2S + c63 * k3S + c64 * k4S + c65 * k5S)
        tmpE = E + h * (c61 * k1E + 2.0 * k2E + c63 * k3E + c64 * k4E + c65 * k5E)
        tmpI = I + h * (c61 * k1I + 2.0 * k2I + c63 * k3I + c64 * k4I + c65 * k5I)
        tmpR = R + h * (c61 * k1R + 2.0 * k2R + c63 * k3R + c64 * k4R + c65 * k5R)
        bSI = beta * tmpS * tmpI
        k6S = -bSI + omega * tmpR
        k6E = bSI - sigma * tmpE
        k6I = sigma * tmpE - gamma * tmpI
        k6R = gamma * tmpI - omega * tmpR
        
        # 5th and 4th order coefficients
        b51 = 0.11851851851851852  # 16/135
        b53 = 0.5189863547758284  # 6656/12825
        b54 = 0.5061314903420167  # 28561/56430
        b55 = -0.18  # -9/50
        b56 = 0.03636363636363636  # 2/55
        b41 = 0.11574074074074074  # 25/216
        b43 = 0.5489278752436647  # 1408/2565
        b44 = 0.5353313840155945  # 2197/4104
        b45 = -0.2  # -1/5
        
        newS = S + h * (b51 * k1S + b53 * k3S + b54 * k4S + b55 * k5S + b56 * k6S)
        newE = E + h * (b51 * k1E + b53 * k3E + b54 * k4E + b55 * k5E + b56 * k6E)
        newI = I + h * (b51 * k1I + b53 * k3I + b54 * k4I + b55 * k5I + b56 * k6I)
        newR = R + h * (b51 * k1R + b53 * k3R + b54 * k4R + b55 * k5R + b56 * k6R)
        
        y4S = S + h * (b41 * k1S + b43 * k3S + b44 * k4S + b45 * k5S)
        y4E = E + h * (b41 * k1E + b43 * k3E + b44 * k4E + b45 * k5E)
        y4I = I + h * (b41 * k1I + b43 * k3I + b44 * k4I + b45 * k5I)
        y4R = R + h * (b41 * k1R + b43 * k3R + b44 * k4R + b45 * k5R)
        
        # Error estimate
        errS = abs(newS - y4S) / (atol + rtol * max(abs(S), abs(newS)))
        errE = abs(newE - y4E) / (atol + rtol * max(abs(E), abs(newE)))
        errI = abs(newI - y4I) / (atol + rtol * max(abs(I), abs(newI)))
        errR = abs(newR - y4R) / (atol + rtol * max(abs(R), abs(newR)))
        max_err = max(errS, errE, errI, errR)
        
        if max_err < 1.0:
            t = t + h
            S, E, I, R = newS, newE, newI, newR
            if max_err > 1e-10:
                h = h * min(5.0, 0.9 * max_err ** (-0.2))
            else:
                h = h * 5.0
        else:
            h = h * max(0.1, 0.9 * max_err ** (-0.25))
    
    return S, E, I, R

# Warm up JIT compilation
_dummy = integrate_seirs_scalar(0.9, 0.05, 0.03, 0.02, 0.0, 10.0, 0.35, 0.2, 0.1, 0.002, 1e-8, 1e-9)

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        y0 = problem["y0"]
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        
        beta = float(params["beta"])
        sigma = float(params["sigma"])
        gamma = float(params["gamma"])
        omega = float(params["omega"])
        
        S, E, I, R = integrate_seirs_scalar(
            float(y0[0]), float(y0[1]), float(y0[2]), float(y0[3]),
            t0, t1, beta, sigma, gamma, omega, 1e-8, 1e-9
        )
        
        return [S, E, I, R]