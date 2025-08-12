import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solves the LTI simulation problem using scipy.signal.lsim.
        
        :param problem: A dictionary representing the LTI simulation problem.
        :return: A dictionary with key "yout" containing the output signal.
        """
        num = problem["num"]
        den = problem["den"]
        u = problem["u"]
        t = problem["t"]
        
        # Create the LTI system object
        system = signal.lti(num, den)
        
        # Simulate the system response
        tout, yout, xout = signal.lsim(system, u, t)
        
        solution = {"yout": yout.tolist()}
        return solution