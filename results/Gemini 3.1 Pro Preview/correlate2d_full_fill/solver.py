import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: tuple, **kwargs) -> np.ndarray:
        a, b = problem
        a32 = a.astype(np.float32, copy=False)
        b32 = np.ascontiguousarray(b[::-1, ::-1], dtype=np.float32)
        return signal.fftconvolve(a32, b32, mode='full')