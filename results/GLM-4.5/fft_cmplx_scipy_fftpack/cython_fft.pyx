# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

# Import C FFT functions
from libc.math cimport M_PI

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_fftn(np.ndarray[complex, ndim=2] arr):
    """
    Compute 2D FFT using Cython-optimized Cooley-Tukey algorithm.
    This is a simplified implementation for demonstration.
    """
    cdef int m = arr.shape[0]
    cdef int n = arr.shape[1]
    cdef np.ndarray[complex, ndim=2] result = np.empty((m, n), dtype=complex)
    cdef np.ndarray[complex, ndim=1] temp
    
    # FFT along rows
    for i in range(m):
        temp = _fft_1d(arr[i, :])
        result[i, :] = temp
    
    # FFT along columns
    for j in range(n):
        temp = _fft_1d(result[:, j])
        result[:, j] = temp
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[complex, ndim=1] _fft_1d(np.ndarray[complex, ndim=1] x):
    """1D FFT using Cooley-Tukey algorithm"""
    cdef int n = x.shape[0]
    cdef np.ndarray[complex, ndim=1] result
    
    if n <= 1:
        return x.copy()
    
    # Divide
    cdef np.ndarray[complex, ndim=1] even = _fft_1d(x[0::2])
    cdef np.ndarray[complex, ndim=1] odd = _fft_1d(x[1::2])
    
    # Conquer
    cdef np.ndarray[complex, ndim=1] T = np.exp(-2j * M_PI * np.arange(n//2) / n)
    cdef np.ndarray[complex, ndim=1] part1 = even + T * odd
    cdef np.ndarray[complex, ndim=1] part2 = even - T * odd
    
    result = np.concatenate([part1, part2])
    return result