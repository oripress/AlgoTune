import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem, **kwargs):
        """
        Optimized LTI system simulation using scipy.signal.lsim.
        """
        num = problem["num"]
        den = problem["den"]
        u = problem["u"]
        t = problem["t"]
        
        # Create the LTI system object
        system = signal.lti(num, den)
        
        # Simulate the system response
        tout, yout, xout = signal.lsim(system, u, t)
        
        return {"yout": yout.tolist()}