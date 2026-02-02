from __future__ import annotations

from typing import Any

import numpy as np

# Cache hot functions to reduce attribute-lookup overhead in tight budgets.
_EIGH = np.linalg.eigh
_SQRT = np.sqrt
_ZEROS_LIKE = np.zeros_like
_TRIL_INDICES = np.tril_indices

try:
    from scipy.linalg.blas import dsyrk as _dsyrk

    _HAVE_DSYRK = True
except Exception:  # pragma: no cover
    _dsyrk = None
    _HAVE_DSYRK = False

# Cache triangular indices for in-place symmetrization in dsyrk branch.
_TRIL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}

def _tril_cache(n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = _TRIL_CACHE.get(n)
    if idx is None:
        idx = _TRIL_INDICES(n, -1)
        _TRIL_CACHE[n] = idx
    return idx

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        A = np.asarray(problem["A"])
        if A.dtype.kind not in ("f", "c"):
            A = A.astype(np.float64, copy=False)

        w, V = _EIGH(A)

        # w is sorted ascending; keep strictly positive eigenvalues.
        i0 = w.searchsorted(0.0, side="right")
        if i0 >= w.size:
            return {"X": _ZEROS_LIKE(A)}

        wpos = w[i0:]
        Vpos = V[:, i0:]

        # X = (Vpos*sqrt(wpos)) (Vpos*sqrt(wpos))^T
        _SQRT(wpos, out=wpos)
        Vpos *= wpos  # in-place column scaling

        n = A.shape[0]
        if (
            _HAVE_DSYRK
            and n >= 128
            and Vpos.dtype == np.float64
            and Vpos.flags.f_contiguous
        ):
            # dsyrk writes only the requested triangle; fill the other in-place.
            X = _dsyrk(1.0, Vpos, lower=0, trans=0)
            ii, jj = _tril_cache(n)  # lower triangle indices (excluding diagonal)
            X[ii, jj] = X[jj, ii]
            return {"X": X}

        return {"X": Vpos @ Vpos.T}