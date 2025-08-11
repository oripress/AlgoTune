import numpy as np
from scipy.fft import dctn

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute 2D DCT Type I via pocketfft dctn in one shot.
        """
        arr = np.asarray(problem, dtype=float)
        # single-threaded to reduce overhead
        return dctn(arr, type=1, axes=(0, 1), overwrite_x=True, workers=1)

        # 1D / trivial cases
        if n0 == 1 and n1 == 1:
            return arr.copy()
        if n0 == 1:
            v = arr[0]
            m = 2*n1 - 2 if n1 > 1 else 1
            ext = np.empty(m, dtype=arr.dtype)
            ext[:n1] = v
            if n1 > 1:
                ext[n1:] = v[-2:0:-1]
            if _HAS_WORKERS:
                try:
                    F = rfftn(ext, overwrite_x=False, workers=workers)
                except TypeError:
                    F = rfftn(ext)
            else:
                F = rfftn(ext)
            return F[:n1].real.reshape(1, n1)
        if n1 == 1:
            v = arr[:, 0]
            m = 2*n0 - 2 if n0 > 1 else 1
            ext = np.empty(m, dtype=arr.dtype)
            ext[:n0] = v
            if n0 > 1:
                ext[n0:] = v[-2:0:-1]
            if _HAS_WORKERS:
                try:
                    F = rfftn(ext, overwrite_x=False, workers=workers)
                except TypeError:
                    F = rfftn(ext)
            else:
                F = rfftn(ext)
            return F[:n0].real.reshape(n0, 1)

        # 2D symmetric extension
        m0 = 2*n0 - 2
        m1 = 2*n1 - 2
        ext = np.empty((m0, m1), dtype=arr.dtype)
        ext[:n0, :n1] = arr
        ext[n0:, :n1] = arr[-2:0:-1, :]
        ext[:n0, n1:] = arr[:, -2:0:-1]
        ext[n0:, n1:] = arr[-2:0:-1, -2:0:-1]

        # real 2D FFT
        if _HAS_WORKERS:
            try:
                F2 = rfftn(ext, axes=(0, 1), overwrite_x=False, workers=workers)
            except TypeError:
                F2 = rfftn(ext, axes=(0, 1))
        else:
            F2 = rfftn(ext, axes=(0, 1))

        return F2[:n0, :n1].real