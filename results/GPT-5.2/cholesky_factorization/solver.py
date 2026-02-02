import numpy as np

try:
    from numpy.linalg import _umath_linalg as _uml  # type: ignore[attr-defined]

    _CHOL_UFUNC = _uml.cholesky_lo  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _CHOL_UFUNC = None

_CHOL = np.linalg.cholesky

# Cache reusable output buffers by (n, dtype.num) to reduce allocations.
_OUT_CACHE: dict[tuple[int, int], np.ndarray] = {}

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["matrix"]
        if _CHOL_UFUNC is not None:
            n = A.shape[0]
            key = (n, A.dtype.num)
            cache = _OUT_CACHE
            out = cache.get(key)
            if out is None:
                out = np.empty((n, n), dtype=A.dtype, order="F")
                cache[key] = out
            _CHOL_UFUNC(A, out=(out,))
            L = out
        else:
            L = _CHOL(A)
        return {"Cholesky": {"L": L}}