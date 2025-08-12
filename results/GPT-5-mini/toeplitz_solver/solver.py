from typing import Any, Dict, List, Optional, Callable
import numpy as np
import inspect as _inspect

# Fast binding to SciPy's solve_toeplitz when available.
_solve_toeplitz_func: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None
try:
    from scipy.linalg import solve_toeplitz as _scipy_solve_toeplitz  # type: ignore

    _sig = _inspect.signature(_scipy_solve_toeplitz)
    _params = _sig.parameters

    if "check_finite" in _params and "overwrite_b" in _params:
        def _solve_toeplitz_func(c: np.ndarray, r: np.ndarray, b: np.ndarray) -> np.ndarray:
            # avoid copies/checks when supported
            return _scipy_solve_toeplitz((c, r), b, check_finite=False, overwrite_b=True)  # type: ignore
    elif "check_finite" in _params:
        def _solve_toeplitz_func(c: np.ndarray, r: np.ndarray, b: np.ndarray) -> np.ndarray:
            return _scipy_solve_toeplitz((c, r), b, check_finite=False)  # type: ignore
    else:
        def _solve_toeplitz_func(c: np.ndarray, r: np.ndarray, b: np.ndarray) -> np.ndarray:
            return _scipy_solve_toeplitz((c, r), b)  # type: ignore
except Exception:
    _solve_toeplitz_func = None

class Solver:
    """
    Minimal, low-overhead Toeplitz solver. Prefers SciPy's specialized routine and
    falls back to a dense solve if SciPy is not available.
    """

    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        if not isinstance(problem, dict):
            raise ValueError("problem must be a dict with keys 'c', 'r', and 'b'")

        try:
            c = np.asarray(problem["c"], dtype=np.float64)
            r = np.asarray(problem["r"], dtype=np.float64)
            b = np.asarray(problem["b"], dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Invalid problem input: {e}")

        n = b.size
        if n == 0:
            return []

        if c.size != n or r.size != n:
            raise ValueError("'c', 'r', and 'b' must have the same length")

        # Use SciPy's Levinson-style solver if available (usually fastest).
        if _solve_toeplitz_func is not None:
            x = _solve_toeplitz_func(c, r, b)
            return np.asarray(x, dtype=np.float64).tolist()

        # Fallback: build dense Toeplitz matrix and solve via LAPACK.
        idx = np.arange(n)
        diff = idx.reshape(-1, 1) - idx.reshape(1, -1)
        T = np.empty((n, n), dtype=np.float64)
        pos = diff >= 0
        T[pos] = c[diff[pos]]
        T[~pos] = r[-diff[~pos]]
        x = np.linalg.solve(T, b)
        return np.asarray(x, dtype=np.float64).tolist()