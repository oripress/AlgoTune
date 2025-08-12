import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem):
        """
        Compute the output of a continuous‑time LTI system using scipy.signal.lsim.
        This implementation directly uses the reference solver to ensure numerical
        correctness.
        """
        # Extract problem data
        num = problem["num"]
        den = problem["den"]
        u = np.asarray(problem["u"], dtype=float)
        t = np.asarray(problem["t"], dtype=float)

        # Use accurate continuous‑time simulation for all cases
        _, yout, _ = signal.lsim((num, den), u, t)
        yout = np.squeeze(yout)
        return {"yout": yout.tolist()}