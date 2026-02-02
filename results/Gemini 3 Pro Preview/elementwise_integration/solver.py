from typing import Any
import numpy as np
from numba import njit
import math

@njit(fastmath=True)
def rgamma(z):
    if z >= 0.0:
        if z == 0.0: 
            return 0.0
        if z > 171.0:
            return 0.0
        return 1.0 / math.gamma(z)
    else:
        # Check for negative integers to avoid numerical noise
        # Using a small epsilon for float comparison
        if abs(z - round(z)) < 1e-15:
            return 0.0
        # Reflection formula: 1/Gamma(z) = sin(pi*z)/pi * Gamma(1-z)
        return (math.sin(math.pi * z) / math.pi) * math.gamma(1.0 - z)

@njit(fastmath=True)
def wright_bessel_scalar(a, b, x):
    # Term k=0: 1/Gamma(b)
    res = rgamma(b)
    
    if x == 0.0:
        return res
        
    x_pow = 1.0
    fact = 1.0
    
    # Loop for series summation
    # 150 iterations should be enough for convergence
    for k in range(1, 150):
        x_pow *= x
        fact *= k
        
        arg = a * k + b
        term = (x_pow / fact) * rgamma(arg)
        
        res += term
        
        # Convergence check
        if abs(term) < 1e-16 * abs(res):
            break
            
    return res

@njit(fastmath=True)
def solve_numba(a, b, lower, upper, out):
    n = a.shape[0]
    for i in range(n):
        ai = a[i]
        bi = b[i]
        low = lower[i]
        up = upper[i]
        
        b_prime = bi - ai
        
        val_up = wright_bessel_scalar(ai, b_prime, up)
        val_low = wright_bessel_scalar(ai, b_prime, low)
        
        out[i] = val_up - val_low

class Solver:
    def __init__(self):
        # Trigger JIT compilation with dummy data
        a = np.array([1.0], dtype=np.float64)
        b = np.array([1.0], dtype=np.float64)
        l = np.array([0.0], dtype=np.float64)
        u = np.array([1.0], dtype=np.float64)
        out = np.empty(1, dtype=np.float64)
        solve_numba(a, b, l, u, out)

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        a = np.array(problem["a"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        lower = np.array(problem["lower"], dtype=np.float64)
        upper = np.array(problem["upper"], dtype=np.float64)
        
        n = len(a)
        res = np.empty(n, dtype=np.float64)
        
        solve_numba(a, b, lower, upper, res)
        
        return {"result": res.tolist()}