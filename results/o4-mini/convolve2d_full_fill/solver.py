import os
import numpy as np
# Try to import next_fast_len from numpy or scipy for optimal FFT sizes
try:
    from numpy.fft import rfft2, irfft2, next_fast_len
except ImportError:
    from numpy.fft import rfft2, irfft2
    from scipy.fftpack import next_fast_len

class Solver:
    def solve(self, problem, **kwargs):
        a, b = problem
        # ensure C-contiguous
        a = np.ascontiguousarray(a)
        b = np.ascontiguousarray(b)
        # output full convolution shape
        out_h = a.shape[0] + b.shape[0] - 1
        out_w = a.shape[1] + b.shape[1] - 1
        # determine FFT shapes that are fast
        fft_h = next_fast_len(out_h)
        fft_w = next_fast_len(out_w)
        # cast to float32 for faster FFT, compute transforms
        a32 = a.astype(np.float32)
        b32 = b.astype(np.float32)
        fa = rfft2(a32, s=(fft_h, fft_w))
        fb = rfft2(b32, s=(fft_h, fft_w))
        # multiply in freq domain and inverse transform
        conv32 = irfft2(fa * fb, s=(fft_h, fft_w))
        # slice to desired output size and cast back to float64
        result = conv32[:out_h, :out_w].astype(np.float64)
        return result