import numpy as np
from scipy.integrate import tanhsinh
from scipy.special import gamma, gammaln

class Solver:
    def solve(self, problem, **kwargs):
        a = np.array(problem["a"])
        b = np.array(problem["b"])
        lower = np.array(problem["lower"])
        upper = np.array(problem["upper"])
        
        # Preallocate result array
        integrals = np.zeros_like(a)
        
        # Vectorized integration using tanhsinh
        for i in range(len(a)):
            res = tanhsinh(
                lambda x: self.wright_bessel(a[i], b[i], x),
                lower[i], 
                upper[i]
            )
            integrals[i] = res.integral
            
        return {"result": integrals.tolist()}
    
    def wright_bessel(self, a, b, x):
        """Compute Wright's Bessel function using series expansion."""
        if x == 0:
            return 1.0 / gamma(b)
        
        # Use logarithms for numerical stability
        log_x = np.log(x)
        res = 0.0
        max_abs_term = 0.0
        tol = 1e-12
        
        for k in range(2000):
            # Compute term in log space
            log_term = k * log_x - gammaln(k+1) - gammaln(a*k + b)
            term = np.exp(log_term)
            res += term
            
            # Track maximum term magnitude for convergence
            abs_term = abs(term)
            if abs_term > max_abs_term:
                max_abs_term = abs_term
            
            # Check convergence relative to largest term
            if abs_term < tol * max_abs_term:
                break
                
        return res