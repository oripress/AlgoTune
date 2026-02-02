import numpy as np
import scipy.optimize

class Solver:
    def __init__(self):
        # Fixed parameters (based on typical diode equation values)
        self.a2 = 1e-12
        self.a3 = 1.0
        self.a4 = 1e9
        self.a5 = 0.026
        
    def func(self, x, a0, a1, a2, a3, a4, a5):
        """f(x) = a1 - a2*(exp((a0+x*a3)/a5) - 1) - (a0+x*a3)/a4 - x"""
        z = (a0 + x * a3) / a5
        z = np.clip(z, -700, 700)
        return a1 - a2 * (np.exp(z) - 1) - (a0 + x * a3) / a4 - x
    
    def fprime(self, x, a0, a1, a2, a3, a4, a5):
        """Derivative of f with respect to x"""
        z = (a0 + x * a3) / a5
        z = np.clip(z, -700, 700)
        return -a2 * (a3 / a5) * np.exp(z) - a3 / a4 - 1
        
    def solve(self, problem, **kwargs):
        """
        Vectorized Newton-Raphson solver
        """
        x0_arr = np.array(problem["x0"], dtype=np.float64)
        a0_arr = np.array(problem["a0"], dtype=np.float64)
        a1_arr = np.array(problem["a1"], dtype=np.float64)
        
        n = len(x0_arr)
        args = (a0_arr, a1_arr, self.a2, self.a3, self.a4, self.a5)
        
        try:
            roots_arr = scipy.optimize.newton(self.func, x0_arr, fprime=self.fprime, args=args, maxiter=100, tol=1.48e-8)
            if np.isscalar(roots_arr):
                roots_arr = np.array([roots_arr])
        except Exception:
            roots_arr = np.full(n, np.nan)
        
        return {"roots": roots_arr}