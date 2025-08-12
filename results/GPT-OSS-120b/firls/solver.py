import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem, **kwargs):
        """
        Design a linear-phase FIR filter using least-squares (SciPy's firls).

        Parameters
        ----------
        problem : tuple[int, tuple[float, float]]
            n (int): half filter length; the actual filter length is 2*n+1 (odd).
            edges (tuple[float, float]): normalized frequency band edges (0 < f1 < f2 < 1).

        Returns
        -------
        np.ndarray
            FIR filter coefficients of length 2*n+1.
        """
        n, edges = problem
        N = 2 * n + 1  # actual odd filter length
        # Ensure edges is a tuple (JSON may convert to list)
        edges = tuple(edges)
        # SciPy's firls expects band edges including 0 and 1, and corresponding
        # desired amplitudes. The reference uses weights [1,1,0,0] to specify
        # passband (amp=1) and stopband (amp=0) with equal importance.
        coeffs = signal.firls(N, (0.0, *edges, 1.0), [1, 1, 0, 0])
        return coeffs