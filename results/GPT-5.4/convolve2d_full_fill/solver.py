import numpy as np
from scipy import signal
from scipy.fft import irfftn, next_fast_len, rfftn

class Solver:
    def __init__(self):
        self._workers = 4
        self._direct_threshold = 2_000_000

    def solve(self, problem, **kwargs):
        a, b = problem
        a = np.asarray(a)
        b = np.asarray(b)

        if a.size * b.size <= self._direct_threshold:
            return signal.convolve2d(a, b, mode="full", boundary="fill")

        out0 = a.shape[0] + b.shape[0] - 1
        out1 = a.shape[1] + b.shape[1] - 1
        fft0 = next_fast_len(out0, real=True)
        fft1 = next_fast_len(out1, real=True)

        fa = rfftn(a, (fft0, fft1), workers=self._workers)
        fa *= rfftn(b, (fft0, fft1), workers=self._workers)
        result = irfftn(fa, (fft0, fft1), workers=self._workers, overwrite_x=True)

        if fft0 != out0:
            result = result[:out0]
        if fft1 != out1:
            result = result[:, :out1]
        return result