import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem, **kwargs):
        x = problem["signal_x"]
        y = problem["signal_y"]
        mode = problem.get("mode", "full")
        
        lx = len(x)
        ly = len(y)
        
        if lx == 0 or ly == 0:
            return {"convolution": []}

        # Heuristic threshold for direct convolution
        if lx * ly < 2000:
            if mode == "same":
                out = np.convolve(x, y, mode="full")
                full_len = len(out)
                start = (full_len - lx) // 2
                result = out[start : start + lx]
                return {"convolution": result.tolist()}
            else:
                result = np.convolve(x, y, mode=mode)
                return {"convolution": result.tolist()}
        
        # Pass lists directly to fftconvolve
        result = signal.fftconvolve(x, y, mode=mode)
        return {"convolution": result.tolist()}