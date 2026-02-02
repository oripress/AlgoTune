from __future__ import annotations

from typing import Any

import numpy as np

try:
    # Direct import avoids module attribute lookup overhead in the hot path.
    from scipy.fft import fftn as _fftn  # type: ignore
except Exception:  # pragma: no cover
    _fftn = None

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        a = np.asarray(problem)

        # Avoid internal copies where possible.
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a)

        fftn = _fftn
        if fftn is None:
            return np.fft.fftn(a)

        # Threading heuristic: only for sufficiently large transforms.
        workers = -1 if a.size >= 256 * 256 else 1

        # overwrite_x=True can reduce copies (input is not required to be preserved).
        return fftn(a, workers=workers, overwrite_x=True)