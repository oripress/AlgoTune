import numpy as np
from scipy.signal import lti, lsim
from numba import jit

class Solver:
    def solve(self, problem):
        """Solve the LTI system simulation with JIT compilation."""
        # Use scipy.signal.lsim for accurate results
        system = lti(problem["num"], problem["den"])
        _, yout, _ = lsim(system, problem["u"], problem["t"])
        return {"yout": yout.tolist()}