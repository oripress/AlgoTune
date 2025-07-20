import numpy as np
from scipy import signal
from numba import njit

# Optimized direct convolution with improved memory access and vectorization
@njit(fastmath=True, cache=True)
def direct_convolve(a, b, mode):
    n = len(a)
    m = len(b)
    
    if mode == "full":
        out_size = n + m - 1
        result = np.zeros(out_size)
        
        # Vectorized convolution
        for j in range(m):
            start = j
            end = j + n
            # Vectorized multiplication
            result[start:end] += a * b[j]
        return result
        
    else:  # valid mode
        if n < m:
            return np.array([])
        out_size = n - m + 1
        result = np.zeros(out_size)
        
        # Optimized valid convolution
        for i in range(out_size):
            result[i] = np.dot(a[i:i+m], b)
        return result

# Improved FFT convolution with optimized padding
@njit(fastmath=True, cache=True)
def fft_convolve(a, b, mode):
    n = len(a)
    m = len(b)
    
    if mode == "full":
        out_size = n + m - 1
        # Efficient padding using next power of 2
        fft_size = 1
        while fft_size < out_size:
            fft_size *= 2
            
        # Perform FFT convolution
        a_fft = np.fft.rfft(a, fft_size)
        b_fft = np.fft.rfft(b, fft_size)
        result = np.fft.irfft(a_fft * b_fft, fft_size)
        return result[:out_size]
        
    else:  # valid mode
        if n < m:
            return np.array([])
        out_size = n - m + 1
        # Efficient padding using next power of 2
        fft_size = 1
        while fft_size < n:
            fft_size *= 2
            
        # Perform FFT convolution
        a_fft = np.fft.rfft(a, fft_size)
        b_fft = np.fft.rfft(b, fft_size)
        result = np.fft.irfft(a_fft * b_fft, fft_size)
        return result[m-1:m-1+out_size]

def process_pair(a, b, mode):
    # Handle empty arrays
    if len(a) == 0 or len(b) == 0:
        return np.array([])
    
    # Special cases for very small arrays
    if len(b) == 1:
        return a * b[0] if mode == "full" else a[:len(a)] * b[0]
    if len(a) == 1:
        return b * a[0] if mode == "full" else np.array([])
    
    n = len(a)
    m = len(b)
    product_size = n * m
    
    # Use optimized direct convolution for very small problems
    if product_size < 128:
        return direct_convolve(a, b, mode)
    
    # Use our FFT convolution for medium problems
    if product_size < 4096:
        return fft_convolve(a, b, mode)
    
    # Use scipy's optimized convolution for large problems
    return signal.fftconvolve(a, b, mode=mode)

class Solver:
    def __init__(self, mode='full'):
        self.mode = mode
        
    def solve(self, problem, **kwargs):
        # Handle both single pair and list of pairs
        if isinstance(problem, tuple) and len(problem) == 2:
            a, b = problem
            return process_pair(a, b, self.mode)
        elif isinstance(problem, list):
            return [process_pair(a, b, self.mode) for a, b in problem]
        else:
            raise ValueError("Input must be a tuple (single pair) or list of pairs")