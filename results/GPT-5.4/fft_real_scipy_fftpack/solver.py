import numpy as np
from scipy.fftpack import fft2 as _fft2

_BASE = np.arange(16, dtype=np.float64)
_DFT = {
    n: np.exp((-2j * np.pi / n) * np.outer(_BASE[:n], _BASE[:n]))
    for n in range(3, 17)
    if n != 4
}

class Solver:
    def solve(self, problem, **kwargs):
        arr = problem if isinstance(problem, np.ndarray) else np.asarray(problem)
        n0 = arr.shape[0]
        n1 = arr.shape[1]

        if n0 == 1 and n1 == 1:
            return arr.astype(np.complex128, copy=False)

        if n0 == 2 and n1 == 2:
            a00 = arr[0, 0]
            a01 = arr[0, 1]
            a10 = arr[1, 0]
            a11 = arr[1, 1]
            s0 = a00 + a01
            d0 = a00 - a01
            s1 = a10 + a11
            d1 = a10 - a11
            out = np.empty((2, 2), dtype=np.complex128)
            out[0, 0] = s0 + s1
            out[0, 1] = d0 + d1
            out[1, 0] = s0 - s1
            out[1, 1] = d0 - d1
            return out

        if n0 == 4 and n1 == 4:
            t = np.empty((4, 4), dtype=np.complex128)
            a = arr[:, 0]
            b = arr[:, 1]
            c = arr[:, 2]
            d = arr[:, 3]
            ac = a - c
            t[:, 0] = a + b + c + d
            t[:, 1] = ac + 1j * (d - b)
            t[:, 2] = a - b + c - d
            t[:, 3] = ac + 1j * (b - d)

            out = np.empty((4, 4), dtype=np.complex128)
            a = t[0, :]
            b = t[1, :]
            c = t[2, :]
            d = t[3, :]
            ac = a - c
            out[0, :] = a + b + c + d
            out[1, :] = ac + 1j * (d - b)
            out[2, :] = a - b + c - d
            out[3, :] = ac + 1j * (b - d)
            return out

        if n0 == n1 and n0 <= 16:
            dft = _DFT[n0]
            return dft @ arr @ dft

        return _fft2(arr)