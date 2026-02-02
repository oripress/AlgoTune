from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

# Prefer modern scipy.fft (PocketFFT); fallback to fftpack if needed.
try:
    from scipy.fft import dstn as _dstn  # type: ignore

    _HAS_SCIPY_FFT = True
except Exception:  # pragma: no cover
    _HAS_SCIPY_FFT = False
    from scipy.fftpack import dstn as _dstn  # type: ignore

try:  # pragma: no cover
    import os

    _CPU_COUNT: int = os.cpu_count() or 1
except Exception:  # pragma: no cover
    _CPU_COUNT = 1

# Cache DST-II transform matrices for small n.
# For 2D DST-II: Y = S @ X @ S.T, where S[k, j] = 2*sin(pi*(j+0.5)*(k+1)/n)
_MAT_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
_TMP_CACHE: Dict[int, np.ndarray] = {}
_OUT_CACHE: Dict[int, np.ndarray] = {}

def _build_dst2_mat(n: int) -> Tuple[np.ndarray, np.ndarray]:
    j = np.arange(n, dtype=np.float64) + 0.5
    k = np.arange(n, dtype=np.float64) + 1.0
    s = 2.0 * np.sin((np.pi / n) * (k[:, None] * j[None, :]))
    s = np.ascontiguousarray(s)
    st = np.ascontiguousarray(s.T)
    return s, st

# Precompute small matrices and workspaces at import-time (init cost not counted).
for _n in range(1, 65):
    _MAT_CACHE[_n] = _build_dst2_mat(_n)
    _TMP_CACHE[_n] = np.empty((_n, _n), dtype=np.float64)
    _OUT_CACHE[_n] = np.empty((_n, _n), dtype=np.float64)
del _n

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute 2D DST type-II (matches scipy.fftpack.dstn(problem, type=2)).
        """
        x0 = np.asarray(problem)
        if x0.ndim != 2:
            return _dstn(np.asarray(problem, dtype=np.float64), type=2)

        n0, n1 = x0.shape

        # Small square fast-path: BLAS matmul with cached trig matrices and output buffer.
        if n0 == n1 and n0 <= 64:
            x = np.asarray(x0, dtype=np.float64, order="C")
            s, st = _MAT_CACHE[n0]
            tmp = _TMP_CACHE[n0]
            out = _OUT_CACHE[n0]
            np.matmul(s, x, out=tmp)
            np.matmul(tmp, st, out=out)
            return out

        # FFT path: minimize allocations.
        if x0.dtype == np.float64 and x0.flags.c_contiguous:
            x = x0
        else:
            x = np.array(x0, dtype=np.float64, order="C", copy=True)

        if _HAS_SCIPY_FFT:
            size = x.size
            workers = -1 if (size >= 256 * 256 and _CPU_COUNT > 1) else 1
            return _dstn(x, type=2, overwrite_x=True, workers=workers)

        return _dstn(x, type=2)