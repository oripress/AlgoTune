import numpy as np
import scipy.optimize

class Solver:
    def __init__(self):
        # Fixed parameters from the problem description
        self.a2 = 1.0
        self.a3 = 1.0
        self.a4 = 1.0
        self.a5 = 1.0
    
    def func(self, x, a0, a1, a2, a3, a4, a5):
        """The function f(x, a0..a5) = a1 - a2*(exp((a0+x*a3)/a5) - 1) - (a0+x*a3)/a4 - x"""
        return a1 - a2 * (np.exp((a0 + x * a3) / a5) - 1) - (a0 + x * a3) / a4 - x
    
    def fprime(self, x, a0, a1, a2, a3, a4, a5):
        """Derivative of f with respect to x"""
        return -a2 * a3 / a5 * np.exp((a0 + x * a3) / a5) - a3 / a4 - 1
    
    def solve(self, problem: dict[str, list[float]]) -> dict[str, list[float]]:
        """
        Finds roots using a single vectorized call to scipy.optimize.newton.
        """
        try:
            x0_arr = np.array(problem["x0"])
            a0_arr = np.array(problem["a0"])
            a1_arr = np.array(problem["a1"])
            n = len(x0_arr)
            if len(a0_arr) != n or len(a1_arr) != n:
                raise ValueError("Input arrays have mismatched lengths")
        except Exception:
            return {"roots": []}
        
        # Assemble args tuple for vectorized function
        args = (a0_arr, a1_arr, self.a2, self.a3, self.a4, self.a5)
        
        try:
            # Perform vectorized call
            roots_arr = scipy.optimize.newton(self.func, x0_arr, fprime=self.fprime, args=args)
            
            # Check if newton returned a scalar unexpectedly
            if np.isscalar(roots_arr):
                roots_arr = np.array([roots_arr])
            
            # Pad with NaN if output length doesn't match input
            if len(roots_arr) != n:
                roots_list = list(roots_arr) + [float("nan")] * (n - len(roots_arr))
            else:
                roots_list = roots_arr
                
        except RuntimeError:
            roots_list = [float("nan")] * n
        except Exception:
            roots_list = [float("nan")] * n
        
        return {"roots": roots_list}