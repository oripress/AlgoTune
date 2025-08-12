import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem, **kwargs):
        # Extract problem data with optimized array conversion
        num = np.asarray(problem["num"], dtype=np.float64)
        den = np.asarray(problem["den"], dtype=np.float64)
        u = np.asarray(problem["u"], dtype=np.float64)
        t = np.asarray(problem["t"], dtype=np.float64)
        
        # Convert to state-space form
        A, B, C, D = signal.tf2ss(num, den)
        
        # Create state-space system
        system = signal.StateSpace(A, B, C, D)
        
        # Simulate the system response
        tout, yout, xout = signal.lsim(system, u, t)
        
        # Return the output as list
        return {"yout": yout.tolist()}