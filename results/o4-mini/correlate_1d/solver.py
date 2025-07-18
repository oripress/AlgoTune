import numpy as np
from scipy.signal import fftconvolve
from numpy.lib.stride_tricks import sliding_window_view

class Solver:
    def __init__(self, mode="full"):
        self.mode = mode

    def solve(self, problem, **kwargs):
        mode = kwargs.get("mode", self.mode)
        self.mode = mode
        results = []
        # threshold on product to choose BLAS-window vs FFT
        window_thr = 1_000_000
        for a, b in problem:
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            na, nb = a.size, b.size
            if mode == "valid" and nb > na:
                continue
            b_rev = b[::-1]
            prod = na * nb
            if prod <= window_thr:
                # BLAS via sliding-window + dot
                if mode == "full":
                    pad = nb - 1
                    a_pad = np.concatenate([
                        np.zeros(pad, dtype=a.dtype),
                        a,
                        np.zeros(pad, dtype=a.dtype)
                    ])
                    windows = sliding_window_view(a_pad, window_shape=nb, axis=0)
                else:
                    windows = sliding_window_view(a, window_shape=nb, axis=0)
                res = windows.dot(b_rev)
            else:
                # FFT-based correlation as convolution
                conv = fftconvolve(a, b_rev, mode="full")
                if mode == "valid":
                    res = conv[nb - 1 : na]
                else:
                    res = conv
            results.append(res)
        return results