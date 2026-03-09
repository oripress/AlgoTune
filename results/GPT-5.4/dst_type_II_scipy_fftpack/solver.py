import os
from typing import Any

import numpy as np
import scipy.fft
import scipy.fft._pocketfft as _pocketfft

class Solver:
    def __init__(self) -> None:
        self._dst = _pocketfft.dst
        self._dstn = _pocketfft.dstn
        self._workers = max(1, os.cpu_count() or 1)

        max_small_n = 24
        mats: list[np.ndarray | None] = [None] * (max_small_n + 1)
        mats_t: list[np.ndarray | None] = [None] * (max_small_n + 1)
        for n in range(1, max_small_n + 1):
            k = np.arange(1, n + 1, dtype=np.float64)[:, None]
            j = (2 * np.arange(n, dtype=np.float64) + 1.0)[None, :]
            mat = np.ascontiguousarray(2.0 * np.sin((np.pi / (2.0 * n)) * (k * j)))
            mats[n] = mat
            mats_t[n] = np.ascontiguousarray(mat.T)
        self._small_mats = mats
        self._small_mats_t = mats_t
        self._max_small_n = max_small_n

    def solve(self, problem, **kwargs) -> Any:
        a = np.asarray(problem, dtype=np.float64)
        if a.ndim != 2:
            return self._dstn(a, type=2)

        m, n = a.shape
        if m == n and n <= self._max_small_n:
            s = self._small_mats[n]
            st = self._small_mats_t[n]
            return s @ a @ st

        workers = 1
        tmp = self._dst(a, type=2, axis=1, workers=workers)
        return self._dst(tmp, type=2, axis=0, workers=workers, overwrite_x=True)