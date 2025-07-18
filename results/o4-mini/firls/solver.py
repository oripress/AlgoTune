import numpy as np
from scipy.signal import firls

class Solver:
    def solve(self, problem, **kwargs) -> np.ndarray:
        """
        Design a linear-phase FIR filter of odd length N = 2*M+1
        that approximates the desired frequency response in a
        least-squares sense, matching scipy.signal.firls behavior.
        problem: (half-length M, (edge1, edge2))
        """
        M0, edges = problem
        M = int(M0)
        # total number of taps (odd)
        N = 2 * M + 1
        # ensure edges are floats
        e1 = float(edges[0])
        e2 = float(edges[1])
        # band edges [0, e1, e2, 1]
        bands = [0.0, e1, e2, 1.0]
        # desired amplitude [1,1,0,0]
        desired = [1.0, 1.0, 0.0, 0.0]
        # compute FIR least-squares coefficients
        h = firls(N, bands, desired)
        return h