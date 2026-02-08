import numpy as np
import scipy.fft as _fft
from numba import njit, prange

@njit(cache=True, parallel=True, fastmath=True)
def dct1_1d_rows(x, n_minus_1):
    """Apply DCT-I along axis=1 (rows) using direct computation."""
    rows, cols = x.shape
    result = np.empty((rows, cols), dtype=np.float64)
    
    if n_minus_1 == 0:
        for i in prange(rows):
            result[i, 0] = x[i, 0]
        return result
    
    inv_n = np.pi / n_minus_1
    
    for i in prange(rows):
        for k in range(cols):
            s = 0.0
            for j in range(cols):
                s += x[i, j] * np.cos(inv_n * k * j)
            result[i, k] = s
    
    return result

@njit(cache=True, parallel=True, fastmath=True)
def dct1_1d_cols(x, n_minus_1):
    """Apply DCT-I along axis=0 (columns) using direct computation."""
    rows, cols = x.shape
    result = np.empty((rows, cols), dtype=np.float64)
    
    if n_minus_1 == 0:
        for j in prange(cols):
            result[0, j] = x[0, j]
        return result
    
    inv_n = np.pi / n_minus_1
    
    for k in prange(rows):
        for j in range(cols):
            s = 0.0
            for i in range(rows):
                s += x[i, j] * np.cos(inv_n * k * i)
            result[k, j] = s
    
    return result

class Solver:
    def __init__(self):
        # Warm up numba
        dummy = np.zeros((3, 3), dtype=np.float64)
        dct1_1d_rows(dummy, 2)
        dct1_1d_cols(dummy, 2)
    
    def solve(self, problem, **kwargs):
        """Compute the N-dimensional DCT Type I."""
        x = np.asarray(problem, dtype=np.float64)
        rows, cols = x.shape
        
        if rows <= 8 and cols <= 8:
            # Direct computation for small arrays
            temp = dct1_1d_rows(x, cols - 1)
            return dct1_1d_cols(temp, rows - 1)
        else:
            return _fft.dctn(x, type=1, workers=-1)