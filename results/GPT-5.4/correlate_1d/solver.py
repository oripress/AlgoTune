import numpy as np
from scipy.signal import fftconvolve

class Solver:
    __slots__ = ("mode",)

    def __init__(self, mode: str = "full"):
        self.mode = mode

    @staticmethod
    def _use_fft(lena: int, lenb: int):
        return lena * lenb > 262144 and min(lena, lenb) >= 64

    def solve(self, problem, **kwargs):
        mode = kwargs.get("mode", self.mode)
        corr = np.correlate
        fftconv = fftconvolve
        use_fft = self._use_fft

        if mode == "full":
            results = []
            append = results.append
            for a, b in problem:
                lena = len(a)
                lenb = len(b)
                append(fftconv(a, b[::-1], "full") if use_fft(lena, lenb) else corr(a, b, "full"))
            return results

        if mode == "valid":
            results = []
            append = results.append
            for a, b in problem:
                lena = len(a)
                lenb = len(b)
                if lenb <= lena:
                    append(fftconv(a, b[::-1], "valid") if use_fft(lena, lenb) else corr(a, b, "valid"))
            return results

        if mode == "same":
            results = []
            append = results.append
            for a, b in problem:
                lena = len(a)
                lenb = len(b)
                append(fftconv(a, b[::-1], "same") if use_fft(lena, lenb) else corr(a, b, "same"))
            return results

        return [corr(a, b, "full") for a, b in problem]