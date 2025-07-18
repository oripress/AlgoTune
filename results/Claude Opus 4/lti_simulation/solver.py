import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem, **kwargs):
        """
        Optimized LTI simulation using scipy.signal.lsim with minimal overhead.
        """
        # Direct access without conversion if already numpy arrays
        num = problem["num"]
        den = problem["den"]
        u = problem["u"]
        t = problem["t"]
        
        # Create the LTI system object and simulate in one go
        tout, yout, _ = signal.lsim((num, den), u, t)
        
        # Return as list directly
        return {"yout": yout.tolist()}