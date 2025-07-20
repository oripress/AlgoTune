import numpy as np
from scipy import signal
from numba import jit

# Numba implementation of direct convolution for 'full' and 'valid' modes
@jit(nopython=True, cache=True)
def convolve_direct_numba(a, b, mode_id):
    """
    A Numba-jitted function for 1D convolution.
    mode_id: 1 for 'full', 2 for 'valid'.
    """
    n_a, n_b = a.shape[0], b.shape[0]

    if mode_id == 1:  # 'full' mode
        out_len = n_a + n_b - 1
        if out_len < 0:
            out_len = 0
        out = np.zeros(out_len, dtype=np.float64)
        
        # Ensure 'a' is the larger array to optimize loop by having fewer outer 
        # iterations and more inner iterations, which can be better for caching.
        if n_a < n_b:
            a, b = b, a
            n_a, n_b = n_b, n_a
            
        for i in range(n_a):
            for j in range(n_b):
                out[i + j] += a[i] * b[j]
        return out
    
    elif mode_id == 2:  # 'valid' mode
        # Assumes n_a >= n_b, as per problem spec.
        out_len = n_a - n_b + 1
        if out_len < 0:
            out_len = 0
        out = np.zeros(out_len, dtype=np.float64)
        b_rev = b[::-1] # Reverse the kernel for convolution
        for i in range(out_len):
            s = 0.0
            for j in range(n_b):
                s += a[i + j] * b_rev[j]
            out[i] = s
        return out
    
    # This part should not be reached for the specified modes.
    return np.zeros(0, dtype=np.float64)

class Solver:
    def solve(self, problem, **kwargs):
        """
        Computes the 1D convolution of two arrays, choosing the fastest method.
        """
        a, b = problem
        mode = kwargs.get('mode', 'full')
        
        len_a, len_b = len(a), len(b)

        # Determine smaller and larger array lengths for the heuristic
        if len_a < len_b:
            len_small, len_large = len_b, len_a
        else:
            len_small, len_large = len_a, len_b

        # Heuristic to decide whether to use FFT-based convolution.
        # This is inspired by the heuristic in scipy.signal.convolve.
        use_fft = False
        if len_small > 0:
            # This heuristic is tuned based on typical performance of Numba vs FFT.
            # Numba is very fast for small arrays, so we set a relatively high bar for switching to FFT.
            if len_small >= 25 and (len_large > 50000 or len_small * len_large > 35000):
                use_fft = True
        
        if use_fft:
            # For 'valid' mode, fftconvolve requires the first argument to be at least as large
            # as the second. The problem guarantees len(a) >= len(b) in this case.
            # For 'full' mode, the order doesn't matter.
            return signal.fftconvolve(a, b, mode=mode)
        else:
            # Use the fast Numba implementation for smaller arrays.
            if mode == 'full':
                return convolve_direct_numba(a, b, 1)
            elif mode == 'valid':
                # The problem guarantees len(a) >= len(b) for 'valid' mode.
                return convolve_direct_numba(a, b, 2)
        
        # Fallback for safety, though not expected to be reached given the problem constraints.
        return np.convolve(a, b, mode=mode)