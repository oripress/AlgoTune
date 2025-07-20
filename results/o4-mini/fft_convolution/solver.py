import numpy as np
from scipy.fftpack import next_fast_len

# Prebind FFT functions for faster lookup
_rfft = np.fft.rfft
_irfft = np.fft.irfft
_nfl = next_fast_len

class Solver:
    def solve(self, problem, **kwargs):
        # Load signals as float arrays
        x = np.asarray(problem.get("signal_x", []), dtype=float)
        y = np.asarray(problem.get("signal_y", []), dtype=float)
        lx = x.size; ly = y.size

        # Handle empty inputs
        if lx == 0 or ly == 0:
            return {"convolution": []}

        # Compute FFT length using fastest size
        full_len = lx + ly - 1
        nfft = _nfl(full_len)

        # FFT-based convolution
        X = _rfft(x, nfft)
        Y = _rfft(y, nfft)
        C = X * Y
        conv = _irfft(C, nfft)[:full_len]

        # Determine slicing based on mode
        mode = problem.get("mode", "full")
        if mode == "full":
            out = conv
        elif mode == "same":
            # output length must match length of the first signal (signal_x)
            out_len = lx
            start = (full_len - out_len) // 2
            out = conv[start:start + out_len]
        elif mode == "valid":
            out_len = max(lx, ly) - min(lx, ly) + 1
            start = min(lx, ly) - 1
            out = conv[start:start + out_len]
        else:
            out = conv

        # Return Python list
        return {"convolution": out.tolist()}