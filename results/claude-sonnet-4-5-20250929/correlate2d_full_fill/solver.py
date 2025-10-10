import numpy as np
from scipy import signal
from numba import jit

@jit(nopython=True, cache=True, fastmath=True)
def correlate2d_spatial(a, b):
    """
    Fast spatial domain correlation using numba.
    """
    m, n = a.shape
    p, q = b.shape
    
    # Output shape for full mode
    out_m = m + p - 1
    out_n = n + q - 1
    
    result = np.zeros((out_m, out_n), dtype=a.dtype)
    
    # Flip b for correlation
    b_flipped = np.ascontiguousarray(b[::-1, ::-1])
    
    # Perform correlation
    for i in range(out_m):
        for j in range(out_n):
            # Determine the overlapping region
            a_i_start = max(0, i - p + 1)
            a_i_end = min(m, i + 1)
            a_j_start = max(0, j - q + 1)
            a_j_end = min(n, j + 1)
            
            b_i_start = max(0, p - 1 - i)
            b_i_end = b_i_start + (a_i_end - a_i_start)
            b_j_start = max(0, q - 1 - j)
            b_j_end = b_j_start + (a_j_end - a_j_start)
            
            # Compute dot product
            sum_val = 0.0
            for ii in range(a_i_end - a_i_start):
                for jj in range(a_j_end - a_j_start):
                    sum_val += a[a_i_start + ii, a_j_start + jj] * b_flipped[b_i_start + ii, b_j_start + jj]
            result[i, j] = sum_val
    
    return result

class Solver:
    def __init__(self):
        self.mode = 'full'
        self.boundary = 'fill'

    def solve(self, problem: tuple) -> np.ndarray:
        """
        Compute the 2D correlation using optimized FFT with float32.
        """
        a, b = problem
        
        # Convert to float32 for faster computation
        orig_dtype = a.dtype
        a32 = a.astype(np.float32, copy=False)
        b32 = b.astype(np.float32, copy=False)
        
        # Flip b for correlation
        b_flipped = b32[::-1, ::-1]
        
        # Use fftconvolve which is highly optimized
        result = signal.fftconvolve(a32, b_flipped, mode='full')
        
        # Convert back to original dtype
        return result.astype(orig_dtype)