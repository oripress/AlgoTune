import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """Optimized FFT convolution implementation."""
        signal_x = problem["signal_x"]
        signal_y = problem["signal_y"]
        mode = problem.get("mode", "full")

        # Handle empty signals
        if len(signal_x) == 0 or len(signal_y) == 0:
            return {"convolution": []}

        # Convert to numpy arrays
        x = np.array(signal_x, dtype=float)
        y = np.array(signal_y, dtype=float)
        
        # For small signals, use direct convolution for better performance
        if len(x) < 50 and len(y) < 50:
            result = np.convolve(x, y, mode=mode)
        else:
            # Use scipy.signal.fftconvolve for larger signals
            result = signal.fftconvolve(x, y, mode=mode)
        
        return {"convolution": result.tolist()}