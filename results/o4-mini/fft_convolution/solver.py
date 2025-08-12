import numpy as np
from typing import Any, Dict
from scipy.signal import fftconvolve

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, list]:
        """
        Compute convolution via FFT using scipy.signal.fftconvolve, which handles
        'full', 'same', and 'valid' modes correctly and matches the reference output.
        """
        # Extract inputs
        x = np.asarray(problem.get("signal_x", []), dtype=float)
        y = np.asarray(problem.get("signal_y", []), dtype=float)
        mode = problem.get("mode", "full")

        # Handle empty inputs
        if x.size == 0 or y.size == 0:
            return {"convolution": []}

        # Perform convolution
        result = fftconvolve(x, y, mode=mode)

        # Return as a Python list for validation
        return {"convolution": result.tolist()}