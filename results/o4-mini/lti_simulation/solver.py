import numpy as np
from scipy.signal import lti, lsim

class Solver:
    def solve(self, problem, **kwargs):
        """
        Simulate a continuous‐time LTI system using scipy.signal.lsim.
        """
        # Read problem data
        num = np.asarray(problem["num"], dtype=float)
        den = np.asarray(problem["den"], dtype=float)
        u = np.asarray(problem["u"], dtype=float)
        t = np.asarray(problem["t"], dtype=float)

        # Empty time vector
        if t.size == 0:
            return {"yout": []}

        # Build LTI system and simulate
        system = lti(num, den)
        _, yout, _ = lsim(system, u, t)

        return {"yout": yout.tolist()}