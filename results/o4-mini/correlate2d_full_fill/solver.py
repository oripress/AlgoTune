import numpy as np
try:
    from numpy.fft import rfft2, irfft2, next_fast_len
except ImportError:
    from numpy.fft import rfft2, irfft2
    from scipy.fftpack import next_fast_len

class Solver:
    def solve(self, problem, **kwargs):
        """Compute full 2D correlation with zero-fill boundary via FFT"""
        a, b = problem
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        m, n = a.shape
        p, q = b.shape
        # full output dimensions
        out0 = m + p - 1
        out1 = n + q - 1
        # find fast FFT sizes
        fft0 = next_fast_len(out0)
        fft1 = next_fast_len(out1)
        # FFT of inputs
        FA = rfft2(a, s=(fft0, fft1))
        # reverse b for correlation via convolution
        b_rev = b[::-1, ::-1]
        FB_rev = rfft2(b_rev, s=(fft0, fft1))
        # convolution yields correlation
        corr_full = irfft2(FA * FB_rev, s=(fft0, fft1))
        # slice to full output
        return corr_full[:out0, :out1]