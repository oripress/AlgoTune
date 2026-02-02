from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.fftpack import dctn as _fftpack_dctn  # type: ignore
except Exception:  # pragma: no cover
    _fftpack_dctn = None  # type: ignore

try:
    # PocketFFT backend; can be multithreaded via workers=
    from scipy.fft import dctn as _pocket_dctn  # type: ignore
except Exception:  # pragma: no cover
    _pocket_dctn = None  # type: ignore

class Solver:
    def __init__(self) -> None:
        self._pocket_workers: bool = False
        self._pocket_overwrite: bool = False
        self._workers: int = 1

        if _pocket_dctn is not None:
            # Feature-detect optional kwargs once (init time not counted).
            a = np.zeros((2, 2), dtype=np.float64)
            try:
                _pocket_dctn(a, type=1, workers=1)
                self._pocket_workers = True
            except TypeError:
                self._pocket_workers = False
            try:
                _pocket_dctn(a, type=1, overwrite_x=True)
                self._pocket_overwrite = True
            except TypeError:
                self._pocket_overwrite = False

            if self._pocket_workers:
                # Cap threads to avoid oversubscription on very high core counts.
                import os

                self._workers = min(8, os.cpu_count() or 1)

        # Use pocketfft for medium/large arrays; thread only for very large arrays.
        self._pocket_min_size = 64 * 64
        self._pocket_thread_min_size = 256 * 256

    def solve(self, problem, **kwargs) -> Any:
        if _fftpack_dctn is None:
            raise RuntimeError("scipy.fftpack.dctn is required for this task.")

        x = np.asarray(problem)

        # Medium/large path: pocketfft (optionally multithreaded)
        if _pocket_dctn is not None and x.size >= self._pocket_min_size:
            if self._pocket_workers and x.size >= self._pocket_thread_min_size:
                w = self._workers
                if self._pocket_overwrite:
                    return _pocket_dctn(x, type=1, workers=w, overwrite_x=True)
                return _pocket_dctn(x, type=1, workers=w)

            # Single-thread pocketfft to avoid thread overhead on medium sizes
            if self._pocket_overwrite:
                return _pocket_dctn(x, type=1, overwrite_x=True)
            return _pocket_dctn(x, type=1)

        # Small path: minimize Python-side preprocessing; just enable overwrite when possible.
        if x.flags.writeable:
            return _fftpack_dctn(x, type=1, overwrite_x=True)
        return _fftpack_dctn(x, type=1)