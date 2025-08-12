import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute convolution of two 1‑D signals using an efficient FFT approach.

        Parameters
        ----------
        problem : dict
            Contains:
                - "signal_x": list of numbers
                - "signal_y": list of numbers
                - "mode"   : one of "full", "same", "valid" (default "full")

        Returns
        -------
        dict
            {"convolution": <list of floats>}
        """
        x = np.asarray(problem.get("signal_x", []), dtype=float)
        y = np.asarray(problem.get("signal_y", []), dtype=float)
        mode = problem.get("mode", "full")

        # Empty input handling – result length is zero for any mode
        if x.size == 0 or y.size == 0:
            return {"convolution": []}

        # Use SciPy's highly‑optimized FFT convolution
        try:
            from scipy.signal import fftconvolve
        except ImportError:
            # Fallback to NumPy FFT if SciPy is unavailable
            n_full = x.size + y.size - 1
            n_fft = 1 << (n_full - 1).bit_length()
            X = np.fft.rfft(x, n=n_fft)
            Y = np.fft.rfft(y, n=n_fft)
            conv = np.fft.irfft(X * Y, n=n_fft)[:n_full]
        else:
            conv = fftconvolve(x, y, mode="full")

        # Adjust output according to the requested mode
        if mode == "full":
            result = conv
        elif mode == "same":
            # Output length should be len(x), centered
            target_len = x.size
            start = (conv.size - target_len) // 2
            result = conv[start:start + target_len]
        elif mode == "valid":
            # Only positions where the signals fully overlap
            target_len = max(0, max(x.size, y.size) - min(x.size, y.size) + 1)
            if target_len == 0:
                result = np.array([], dtype=float)
            else:
                start = (conv.size - target_len) // 2
                result = conv[start:start + target_len]
        else:
            # Unknown mode – fall back to full
            result = conv

        # Convert to plain Python list for the validation harness
        return {"convolution": result.tolist()}