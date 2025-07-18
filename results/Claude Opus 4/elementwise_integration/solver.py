from typing import Any
import numpy as np
from scipy.integrate import tanhsinh
from scipy.special import gamma
try:
    from scipy.special import wright_bessel as scipy_wright_bessel
    HAS_SCIPY_WRIGHT_BESSEL = True
except ImportError:
    HAS_SCIPY_WRIGHT_BESSEL = False

class Solver:
    def __init__(self):
        """Precompute anything needed for fast computation."""
        pass
    
    def wright_bessel(self, x, a, b):
        """Compute Wright's Bessel function using series expansion."""
        if HAS_SCIPY_WRIGHT_BESSEL:
            return scipy_wright_bessel(x, a, b)
            
        # Φ(a, b; x) = Σ(k=0 to ∞) x^k / (k! * Γ(ak + b))
        if x == 0:
            return 1.0 / gamma(b)
            
        result = 0.0
        term = 1.0 / gamma(b)  # k=0 term
        result = term
        
        # Use series expansion until convergence
        for k in range(1, 200):  # Limit iterations
            term *= x / k
            gamma_val = gamma(a * k + b)
            if gamma_val == 0 or np.isinf(gamma_val):
                break
            current_term = term / gamma_val
            result += current_term
            
            # Check for convergence
            if abs(current_term) < 1e-15 * abs(result):
                break
                
        return result
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Solve integration problem using tanhsinh integration."""
        a = np.array(problem["a"])
        b = np.array(problem["b"]) 
        lower = np.array(problem["lower"])
        upper = np.array(problem["upper"])
        
        # Use tanhsinh for high precision integration
        res = tanhsinh(self.wright_bessel, lower, upper, args=(a, b))
        
        return {"result": res.integral.tolist()}