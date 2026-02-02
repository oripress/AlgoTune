from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    # SciPy is available in the environment used by the reference.
    from scipy.linalg import expm as _scipy_expm  # type: ignore
except Exception:  # pragma: no cover
    _scipy_expm = None

def _expm_2x2(A: np.ndarray) -> np.ndarray:
    """
    Fast 2x2 matrix exponential using a closed form.

    exp(A) = exp(tr/2) * ( cosh(s) I + sinh(s)/s * (A - tr/2 I) )
    where s^2 = ((a-d)/2)^2 + b*c
    """
    a = float(A[0, 0])
    b = float(A[0, 1])
    c = float(A[1, 0])
    d = float(A[1, 1])

    tr2 = 0.5 * (a + d)
    b00 = a - tr2
    b11 = d - tr2

    s2 = (0.5 * (a - d)) ** 2 + b * c

    # Series near zero avoids division by tiny s.
    if abs(s2) < 1e-30:
        e = np.exp(tr2)
        b01 = b
        b10 = c
        b2_00 = b00 * b00 + b01 * b10
        b2_01 = b01 * (b00 + b11)
        b2_10 = b10 * (b00 + b11)
        b2_11 = b10 * b01 + b11 * b11
        out = np.empty((2, 2), dtype=np.float64)
        out[0, 0] = e * (1.0 + b00 + 0.5 * b2_00)
        out[0, 1] = e * (b01 + 0.5 * b2_01)
        out[1, 0] = e * (b10 + 0.5 * b2_10)
        out[1, 1] = e * (1.0 + b11 + 0.5 * b2_11)
        return out

    if s2 > 0.0:
        s = np.sqrt(s2)
        csh = np.cosh(s)
        sh_over_s = np.sinh(s) / s
    else:
        t = np.sqrt(-s2)
        csh = np.cos(t)
        sh_over_s = np.sin(t) / t

    e = np.exp(tr2)
    out = np.empty((2, 2), dtype=np.float64)
    out[0, 0] = e * (csh + sh_over_s * b00)
    out[0, 1] = e * (sh_over_s * b)
    out[1, 0] = e * (sh_over_s * c)
    out[1, 1] = e * (csh + sh_over_s * b11)
    return out

class Solver:
    def __init__(self) -> None:
        expm = _scipy_expm
        if expm is None:  # pragma: no cover
            self._fast_expm: Callable[[np.ndarray], np.ndarray] | None = None
            return

        probe = np.zeros((1, 1), dtype=np.float64)

        # Fastest options first; probe once to avoid per-call try/except.
        try:
            expm(probe, overwrite_a=True, check_finite=False)  # type: ignore[arg-type]
            self._fast_expm = lambda a: expm(  # type: ignore[misc]
                a, overwrite_a=True, check_finite=False
            )
            return
        except TypeError:
            pass

        try:
            expm(probe, overwrite_a=True)  # type: ignore[arg-type]
            self._fast_expm = lambda a: expm(a, overwrite_a=True)  # type: ignore[misc]
            return
        except TypeError:
            pass

        try:
            expm(probe, check_finite=False)  # type: ignore[arg-type]
            self._fast_expm = lambda a: expm(a, check_finite=False)  # type: ignore[misc]
            return
        except TypeError:
            pass

        self._fast_expm = expm
    def solve(self, problem: dict, **kwargs) -> Any:
        A = np.asarray(problem["matrix"], dtype=np.float64)
        n = A.shape[0]

        if n == 1:
            return {"exponential": np.exp(A)}
        if n == 2:
            return {"exponential": _expm_2x2(A)}

        fast_expm = self._fast_expm
        if fast_expm is None:  # pragma: no cover
            # Last-resort fallback (shouldn't happen here).
            w, V = np.linalg.eig(A)
            expA = V @ np.diag(np.exp(w)) @ np.linalg.inv(V)
            expA = np.real_if_close(expA, tol=1000)
            return {"exponential": expA}

        return {"exponential": fast_expm(A)}