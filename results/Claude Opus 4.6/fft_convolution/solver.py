import numpy as np
from scipy.fft import rfft, irfft, next_fast_len

class Solver:
    def __init__(self):
        # Pre-warm scipy fft with various sizes
        for sz in [8, 64, 512, 4096, 16384, 65536, 262144, 1048576]:
            dummy = np.zeros(sz, dtype=np.float64)
            dummy[0] = 1.0
            r = rfft(dummy, workers=-1, overwrite_x=True)
            irfft(r, n=sz, workers=-1, overwrite_x=True)
    
    def solve(self, problem, **kwargs):
        signal_x = problem["signal_x"]
        signal_y = problem["signal_y"]
        mode = problem.get("mode", "full")
        
        len_x = len(signal_x)
        len_y = len(signal_y)
        
        if len_x == 0 or len_y == 0:
            return {"convolution": np.array([])}
        
        # Convert to numpy arrays efficiently
        if isinstance(signal_x, np.ndarray) and signal_x.dtype == np.float64:
            x = signal_x
        else:
            x = np.asarray(signal_x, dtype=np.float64)
        
        if isinstance(signal_y, np.ndarray) and signal_y.dtype == np.float64:
            y = signal_y
        else:
            y = np.asarray(signal_y, dtype=np.float64)
        
        min_len = min(len_x, len_y)
        
        # For very small signals, use numpy's direct convolution
        if min_len <= 100:
            result = np.convolve(x, y, mode=mode)
            return {"convolution": result}
        
        full_len = len_x + len_y - 1
        fft_len = next_fast_len(full_len)
        
        # Compute FFT convolution using real FFT with all CPU cores
        X = rfft(x, n=fft_len, workers=-1, overwrite_x=True)
        np.multiply(X, rfft(y, n=fft_len, workers=-1, overwrite_x=True), out=X)
        result = irfft(X, n=fft_len, workers=-1, overwrite_x=True)
        
        # Apply mode and return
        if mode == "full":
            return {"convolution": result[:full_len]}
        elif mode == "same":
            start = (len_y - 1) // 2
            return {"convolution": result[start:start + len_x]}
        elif mode == "valid":
            start = min(len_x, len_y) - 1
            return {"convolution": result[start:max(len_x, len_y)]}
        return {"convolution": result[:full_len]}