import numpy as np
from scipy import signal
from scipy.signal import fftconvolve

class Solver:
    def __init__(self):
        self.mode = "full"

    def solve(self, problem, **kwargs):
        a, b = problem
        mode = kwargs.get("mode", getattr(self, "mode", "full"))

        a = np.asarray(a)
        b = np.asarray(b)
        if a.ndim != 1:
            a = a.ravel()
        if b.ndim != 1:
            b = b.ravel()

        if a.size == 0 or b.size == 0:
            return signal.convolve(a, b, mode=mode)

        if mode == "full":
            n = a.size
            m = b.size
            if n > 256 and m > 256 and n * m > 600000:
                kind_a = a.dtype.kind
                kind_b = b.dtype.kind
                if (kind_a == "f" or kind_a == "c") and (kind_b == "f" or kind_b == "c"):
                    return fftconvolve(a, b, mode="full")
        return np.convolve(a, b, mode)