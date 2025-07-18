import numpy as np
from scipy import signal
from scipy.fft import set_workers, get_workers

class Solver:
    def __init__(self):
        # Save original worker count for restoration
        self.original_workers = get_workers()
        
    def solve(self, problem, **kwargs):
        # Use all available CPU cores for FFT
        set_workers(-1)
        
        try:
            # Convert inputs to numpy arrays
            x_arr = np.asarray(problem["signal_x"], dtype=np.float64)
            y_arr = np.asarray(problem["signal_y"], dtype=np.float64)
            mode = problem.get("mode", "full")
            
            # Perform convolution using scipy's optimized FFT
            conv_result = signal.fftconvolve(x_arr, y_arr, mode=mode)
            
            return {"convolution": conv_result.tolist()}
        finally:
            # Restore original worker count
            set_workers(self.original_workers)