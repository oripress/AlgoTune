import numpy as np
from numpy.fft import rfft, irfft

class Solver:
    def solve(self, problem, **kwargs):
        """
        Optimized FFT convolution using numpy's rfft for real signals.
        """
        signal_x = np.asarray(problem["signal_x"], dtype=np.float64)
        signal_y = np.asarray(problem["signal_y"], dtype=np.float64)
        mode = problem.get("mode", "full")
        
        len_x = len(signal_x)
        len_y = len(signal_y)
        
        # Handle edge cases
        if len_x == 0 or len_y == 0:
            return {"convolution": []}
        
        # Compute FFT size - for 'full' mode, we need len_x + len_y - 1
        conv_len = len_x + len_y - 1
        
        # Use next power of 2 for faster FFT (optional optimization)
        fft_len = conv_len
        
        # Perform FFT convolution using rfft (optimized for real signals)
        fft_x = rfft(signal_x, n=fft_len)
        fft_y = rfft(signal_y, n=fft_len)
        
        # Multiply in frequency domain
        result_fft = fft_x * fft_y
        
        # Inverse FFT
        result = irfft(result_fft, n=fft_len)
        
        # Truncate to correct length and handle different modes
        result = result[:conv_len]
        
        if mode == "same":
            # Return central part with length max(len_x, len_y)
            max_len = max(len_x, len_y)
            start = (conv_len - max_len) // 2
            result = result[start:start + max_len]
        elif mode == "valid":
            # Return only fully overlapping part
            valid_len = max(0, max(len_x, len_y) - min(len_x, len_y) + 1)
            start = min(len_x, len_y) - 1
            result = result[start:start + valid_len]
        
        return {"convolution": result.tolist()}