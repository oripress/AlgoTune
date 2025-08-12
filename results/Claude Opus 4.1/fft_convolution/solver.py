import numpy as np
from scipy.signal import fftconvolve

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the convolution problem using the Fast Fourier Transform approach.
        
        :param problem: A dictionary representing the convolution problem.
        :return: A dictionary with key "convolution": a list representing the convolution result.
        """
        signal_x = problem["signal_x"]
        signal_y = problem["signal_y"]
        mode = problem.get("mode", "full")
        
        # Handle empty signals early
        if not signal_x or not signal_y:
            return {"convolution": []}
        
        # Convert to numpy arrays only if necessary
        if not isinstance(signal_x, np.ndarray):
            signal_x = np.asarray(signal_x, dtype=np.float64)
        if not isinstance(signal_y, np.ndarray):
            signal_y = np.asarray(signal_y, dtype=np.float64)
        
        # Use scipy's highly optimized fftconvolve
        result = fftconvolve(signal_x, signal_y, mode=mode)
        
        # Convert to list for output
        return {"convolution": result.tolist()}