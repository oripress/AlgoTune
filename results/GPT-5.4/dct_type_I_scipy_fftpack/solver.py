import os

import numpy as np
from numba import njit
from scipy.fft import dct as _dct
from scipy.fft import dctn as _dctn

def _build_dct1_matrix(m: int) -> np.ndarray:
    if m < 2:
        raise ValueError("DCT-I requires axis length at least 2")
    n = m - 1
    scale = np.pi / n
    k = np.arange(m, dtype=np.float64)[:, None]
    j = np.arange(m, dtype=np.float64)[None, :]
    mat = 2.0 * np.cos(scale * (k * j))
    mat[:, 0] = 1.0
    mat[:, -1] = 1.0
    mat[1::2, -1] = -1.0
    return np.ascontiguousarray(mat)

@njit(cache=True, fastmath=True)
def _small_dct1_1d(x, mat):
    m = x.shape[0]
    out = np.empty(m, dtype=np.float64)
    for i in range(m):
        s = 0.0
        for j in range(m):
            s += mat[i, j] * x[j]
        out[i] = s
    return out

@njit(cache=True, fastmath=True)
def _small_dct1_2d(x, mat0, mat1):
    m0, m1 = x.shape
    tmp = np.empty((m0, m1), dtype=np.float64)
    out = np.empty((m0, m1), dtype=np.float64)

    for i in range(m0):
        for j in range(m1):
            s = 0.0
            for k in range(m0):
                s += mat0[i, k] * x[k, j]
            tmp[i, j] = s

    for i in range(m0):
        for j in range(m1):
            s = 0.0
            for k in range(m1):
                s += tmp[i, k] * mat1[j, k]
            out[i, j] = s

    return out

class Solver:
    __slots__ = ("_small_threshold", "_mats", "_workers", "_thread_threshold")

    def __init__(self):
        self._small_threshold = 12
        self._mats = [None] * (self._small_threshold + 1)
        for m in range(2, self._small_threshold + 1):
            self._mats[m] = _build_dct1_matrix(m)

        sample_x = np.zeros((2, 2), dtype=np.float64)
        sample_m = self._mats[2]
        _small_dct1_1d(sample_x[0], sample_m)
        _small_dct1_2d(sample_x, sample_m, sample_m)

        cpu_count = os.cpu_count() or 1
        self._workers = min(cpu_count, 8)
        self._thread_threshold = 1 << 16

    def solve(self, problem, **kwargs):
        if type(problem) is np.ndarray and problem.dtype == np.float64:
            x = problem
        else:
            x = np.asarray(problem, dtype=np.float64)

        if x.ndim == 1:
            m = x.shape[0]
            if 1 < m <= self._small_threshold:
                return _small_dct1_1d(x, self._mats[m])
            workers = self._workers if (self._workers > 1 and x.size >= self._thread_threshold) else 1
            return _dct(x, type=1, norm=None, workers=workers)

        if x.ndim == 2:
            m0, m1 = x.shape
            if 1 < m0 <= self._small_threshold and 1 < m1 <= self._small_threshold:
                return _small_dct1_2d(x, self._mats[m0], self._mats[m1])

            workers = self._workers if (self._workers > 1 and x.size >= self._thread_threshold) else 1
            tmp = _dct(x, type=1, axis=-1, norm=None, workers=workers)
            return _dct(tmp, type=1, axis=-2, norm=None, workers=workers, overwrite_x=True)

        workers = self._workers if (self._workers > 1 and x.size >= self._thread_threshold) else 1
        return _dctn(x, type=1, norm=None, workers=workers)