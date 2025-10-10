import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[float]]:
        """
        Optimized LTI system simulation using scipy.signal.lsim with minimal overhead.
        """
        num = problem["num"]
        den = problem["den"]
        u = problem["u"]
        t = problem["t"]
        
        # Create system and simulate in one go
        system = signal.lti(num, den)
        _, yout, _ = signal.lsim(system, u, t)
        
        return {"yout": yout.tolist()}