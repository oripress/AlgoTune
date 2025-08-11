import numpy as np
from scipy.signal import fftconvolve
import multiprocessing as mp
from scipy.fft import set_workers
import os
import numba
from numba import jit

class Solver:
    def __init__(self):
        self.mode = 'full'
        self.boundary = 'fill'
        # Use all available CPU cores for FFT operations (max 2)
        num_workers = min(mp.cpu_count(), 2)
        set_workers(num_workers)
        # Set environment variables for optimal FFT performance
        os.environ['OMP_NUM_THREADS'] = str(num_workers)
        os.environ['MKL_NUM_THREADS'] = str(num_workers)
        os.environ['OPENBLAS_NUM_THREADS'] = str(num_workers)
        
        # Set numba threading
        numba.set_num_threads(num_workers)
        
        # Pre-compile Numba functions
        self._compile_numba_functions()
        
        # Cache for small kernels to avoid repeated computation
        self._kernel_cache = {}
    
    def _compile_numba_functions(self):
        """Pre-compile Numba functions to avoid compilation overhead during solve"""
        # Dummy data for compilation
        dummy_a = np.random.rand(10, 10).astype(np.float32)
        dummy_b = np.random.rand(3, 3).astype(np.float32)
        self._direct_convolve_numba(dummy_a, dummy_b)
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _direct_convolve_numba(a, b):
        """Direct 2D convolution using Numba for smaller kernels"""
        a_h, a_w = a.shape
        b_h, b_w = b.shape
        out_h = a_h + b_h - 1
        out_w = a_w + b_w - 1
        
        result = np.zeros((out_h, out_w), dtype=a.dtype)
        
        for i in range(out_h):
            for j in range(out_w):
                total = 0.0
                for m in range(b_h):
                    for n in range(b_w):
                        a_i = i - m
                        a_j = j - n
                        if 0 <= a_i < a_h and 0 <= a_j < a_w:
                            total += a[a_i, a_j] * b[m, n]
                result[i, j] = total
        
        return result
    
    def solve(self, problem, **kwargs) -> np.ndarray:
        """
        Compute the 2D convolution of arrays a and b using "full" mode and "fill" boundary.
        
        :param problem: A tuple (a, b) of 2D arrays.
        :return: A 2D array containing the convolution result.
        """
        a, b = problem
        
        # Ensure arrays are contiguous for better performance
        # Use float32 for faster computation while maintaining precision
        a_contiguous = np.ascontiguousarray(a, dtype=np.float32)
        b_contiguous = np.ascontiguousarray(b, dtype=np.float32)
        
        # Choose method based on kernel size
        # For smaller kernels, direct convolution might be faster
        if b_contiguous.size < 64:  # 8x8 or smaller
            # Check if kernel is in cache
            kernel_hash = hash(b_contiguous.tobytes())
            if kernel_hash in self._kernel_cache:
                # Use cached kernel processing
                result = self._direct_convolve_numba(a_contiguous, self._kernel_cache[kernel_hash])
            else:
                # Cache the kernel for future use
                self._kernel_cache[kernel_hash] = b_contiguous.copy()
                result = self._direct_convolve_numba(a_contiguous, b_contiguous)
        else:
            # Use scipy's optimized fftconvolve for larger kernels
            result = fftconvolve(a_contiguous, b_contiguous, mode='full')
        
        return result.astype(np.float64)