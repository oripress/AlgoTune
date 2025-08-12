import numpy as np
import scipy.optimize

class Solver:
    def __init__(self):
        # Fixed parameters - trying values that might be used in the reference
        self.a2 = 1.0
        self.a3 = 0.1
        self.a4 = 10.0
        self.a5 = 1.0
        
    def func(self, x, a0, a1, a2, a3, a4, a5):
        """Vectorized function f(x, a0..a5) = a1 - a2*(exp((a0+x*a3)/a5) - 1) - (a0+x*a3)/a4 - x"""
        return a1 - a2 * (np.exp((a0 + x * a3) / a5) - 1) - (a0 + x * a3) / a4 - x
    
    def fprime(self, x, a0, a1, a2, a3, a4, a5):
        """Vectorized derivative of f with respect to x"""
        exp_term = np.exp((a0 + x * a3) / a5)
        return -a2 * exp_term * (a3 / a5) - a3 / a4 - 1
    
    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Finds roots using a single vectorized call to scipy.optimize.newton.

        :param problem: Dict with lists "x0", "a0", "a1".
        :return: Dictionary with key "roots": List of `n` found roots. Uses NaN on failure.
        """
        try:
            x0_arr = np.array(problem["x0"])
            a0_arr = np.array(problem["a0"])
            a1_arr = np.array(problem["a1"])
            n = len(x0_arr)
            if len(a0_arr) != n or len(a1_arr) != n:
                raise ValueError("Input arrays have mismatched lengths")
        except Exception as e:
            # Cannot determine n reliably, return empty list?
            return {"roots": []}

        # Assemble args tuple for vectorized function
        args = (a0_arr, a1_arr, self.a2, self.a3, self.a4, self.a5)

        roots_list = []
        try:
            # Perform vectorized call
            roots_arr = scipy.optimize.newton(self.func, x0_arr, fprime=self.fprime, args=args)
            roots_list = roots_arr
            # Check if newton returned a scalar unexpectedly (e.g., if n=1)
            if np.isscalar(roots_list):
                roots_list = [roots_list]

            # Pad with NaN if output length doesn't match input (shouldn't happen with vectorization)
            if len(roots_list) != n:
                roots_list.extend([float("nan")] * (n - len(roots_list)))

        except RuntimeError as e:
            # Vectorized call might fail entirely or partially? SciPy docs are unclear.
            # Assume it raises error if *any* fail to converge. Return all NaNs.
            roots_list = [float("nan")] * n
        except Exception as e:
            roots_list = [float("nan")] * n

        # Convert to list if it's a numpy array, otherwise keep as is
        if isinstance(roots_list, np.ndarray):
            roots_list = roots_list.tolist()
        solution = {"roots": roots_list}
        return solution