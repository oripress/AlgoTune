from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None  # type: ignore[assignment]

if njit is not None:

    @njit(cache=True, fastmath=True)
    def _project_active_compact_inplace(y: np.ndarray, max_iter: int) -> int:
        """
        In-place simplex projection using sorting-free active-set compaction.

        Maintains a compact list of active indices. When an index is removed, we set y[i]=0,
        which is safe because theta is nondecreasing and removed indices are guaranteed to be
        zero in the final projection.

        Returns 1 on success, 0 if not converged within max_iter.
        """
        n = y.size
        idx = np.empty(n, np.int32)

        s = 0.0
        for i in range(n):
            idx[i] = i
            s += y[i]

        m = n
        for _ in range(max_iter):
            theta = (s - 1.0) / m

            s2 = 0.0
            m2 = 0
            for j in range(m):
                i = idx[j]
                v = y[i]
                if v > theta:
                    idx[m2] = i
                    s2 += v
                    m2 += 1
                else:
                    y[i] = 0.0

            if m2 == m:
                # Converged: apply final shift only to active indices.
                for j in range(m):
                    i = idx[j]
                    v = y[i] - theta
                    y[i] = v if v > 0.0 else 0.0
                return 1

            if m2 == 0:
                return 0

            s = s2
            m = m2

        return 0

else:
    _project_active_compact_inplace = None  # type: ignore[assignment]

class Solver:
    __slots__ = ("_use_numba",)

    def __init__(self) -> None:
        self._use_numba = _project_active_compact_inplace is not None
        # Precompile (init time is not counted).
        if self._use_numba:
            _project_active_compact_inplace(np.array([0.0, 0.0], dtype=np.float64), 8)  # type: ignore[misc]

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, np.ndarray]:
        y_in = problem["y"]

        # For numpy input, copy to avoid mutating caller data; for lists, asarray allocates anyway.
        if isinstance(y_in, np.ndarray):
            y = np.array(y_in, dtype=np.float64, copy=True).reshape(-1)
        else:
            y = np.asarray(y_in, dtype=np.float64).reshape(-1)

        n = y.size
        if n == 1:
            y[0] = 1.0
            return {"solution": y}

        # For small n, the classic sort method is typically fastest.
        if (not self._use_numba) or n <= 128:
            u = np.sort(y)[::-1]
            s = np.cumsum(u, dtype=np.float64)
            s -= 1.0
            s /= np.arange(1, n + 1, dtype=np.float64)
            r = int(np.count_nonzero(u > s))
            theta = float(s[r - 1])
            y -= theta
            np.maximum(y, 0.0, out=y)
            return {"solution": y}

        # Fast path: sorting-free active-set compaction.
        if _project_active_compact_inplace(y, 64):  # type: ignore[misc]
            return {"solution": y}

        # Fallback (rare): reconstruct original y and use guaranteed sort method.
        # (Needed because the numba routine may have modified y in-place.)
        y2 = np.asarray(y_in, dtype=np.float64).reshape(-1).copy()
        u = np.sort(y2)[::-1]
        s = np.cumsum(u, dtype=np.float64)
        s -= 1.0
        s /= np.arange(1, n + 1, dtype=np.float64)
        r = int(np.count_nonzero(u > s))
        theta = float(s[r - 1])
        y2 -= theta
        np.maximum(y2, 0.0, out=y2)
        return {"solution": y2}