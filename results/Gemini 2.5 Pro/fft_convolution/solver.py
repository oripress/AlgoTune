import numpy as np
from scipy import fft
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Computes the convolution of two signals using FFT.
        This implementation aims to be faster than scipy.signal.fftconvolve
        by manually handling the FFT steps using scipy.fft, which can be
        more efficient.
        """
        signal_x = problem["signal_x"]
        signal_y = problem["signal_y"]
        mode = problem.get("mode", "full")

        # Using np.asarray to avoid copying if already an array
        x = np.asarray(signal_x, dtype=np.float64)
        y = np.asarray(signal_y, dtype=np.float64)

        len_x = x.shape[0]
        len_y = y.shape[0]

        if len_x == 0 or len_y == 0:
            return {"convolution": []}

        # The full convolution length
        n = len_x + len_y - 1

        # Find a fast length for FFT. This is a key optimization.
        fft_len = fft.next_fast_len(n)

        # Use rfft for real-valued signals, which is faster.
        X = fft.rfft(x, n=fft_len)
        Y = fft.rfft(y, n=fft_len)

        # Element-wise multiplication in the frequency domain
        Z = X * Y

        # Inverse FFT to get the full convolution
        conv_full = fft.irfft(Z, n=fft_len)

        # Trim the result based on the specified mode
        if mode == "full":
            result = conv_full[:n]
        elif mode == "same":
            # The validator expects the output length to be len(signal_x)
            out_len = len_x
            start = (n - out_len) // 2
            result = conv_full[start : start + out_len]
        elif mode == "valid":
            out_len = max(len_x, len_y) - min(len_x, len_y) + 1
            start = min(len_x, len_y) - 1
            result = conv_full[start : start + out_len]
        else:
            # According to the problem, mode is one of 'full', 'same', 'valid'.
            # This path should ideally not be taken.
            raise ValueError(f"Unsupported mode: {mode}")

        # The validator expects a list, not a numpy array.
        return {"convolution": result.tolist()}