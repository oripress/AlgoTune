import numpy as np
from numba import njit
import scipy.optimize

class Solver:
    def __init__(self):
        # Fixed parameters for the task - these need to match the reference
        self.a2 = 1.0
        self.a3 = 1.0  
        self.a4 = 1.0
        self.a5 = 1.0
        
    def func(self, x, a0, a1, a2, a3, a4, a5):
        """Function: f(x, a0..a5) = a1 - a2*(exp((a0+x*a3)/a5) - 1) - (a0+x*a3)/a4 - x"""
        term = (a0 + x * a3) / a5
        return a1 - a2 * (np.exp(term) - 1) - (a0 + x * a3) / a4 - x
    
    def fprime(self, x, a0, a1, a2, a3, a4, a5):
        """Derivative: f'(x) = -a2*a3/a5*exp((a0+x*a3)/a5) - a3/a4 - 1"""
        term = (a0 + x * a3) / a5
        return -a2 * a3 / a5 * np.exp(term) - a3 / a4 - 1
    
    def solve(self, problem):
        """
        Finds roots using scipy.optimize.newton with optimizations.
        
        :param problem: Dict with lists "x0", "a0", "a1".
        :return: Dictionary with key "roots": List of n found roots.
        """
        x0 = np.asarray(problem["x0"], dtype=np.float64)
        a0 = np.asarray(problem["a0"], dtype=np.float64)
        a1 = np.asarray(problem["a1"], dtype=np.float64)
        
        n = len(x0)
        
        # Assemble args tuple for vectorized function
        args = (a0, a1, self.a2, self.a3, self.a4, self.a5)
        
        try:
            # Use scipy.optimize.newton with vectorized functions
            roots = scipy.optimize.newton(
                self.func, x0, fprime=self.fprime, args=args,
                tol=1e-10, maxiter=50
            )
            
            # Handle scalar return for single element
            if np.isscalar(roots):
                roots = np.array([roots])
            
            return {"roots": roots.tolist()}
            
        except Exception as e:
            # Return NaN values on failure
            return {"roots": [float("nan")] * n}