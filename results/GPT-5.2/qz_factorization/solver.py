from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.linalg import qz as _qz
except Exception:  # pragma: no cover
    _qz = None  # type: ignore[assignment]

# Try to access LAPACK directly (less wrapper overhead than scipy.linalg.qz).
try:
    from scipy.linalg.lapack import dgges as _dgges
    from scipy.linalg.lapack import zgges as _zgges
except Exception:  # pragma: no cover
    _dgges = None  # type: ignore[assignment]
    _zgges = None  # type: ignore[assignment]

class Solver:
    __slots__ = ()

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        A_in = problem["A"]
        B_in = problem["B"]

        # Fast path: 1x1 case has a trivial valid QZ with Q=Z=I.
        if len(A_in) == 1:
            a00 = A_in[0][0]
            b00 = B_in[0][0]
            AA = np.array([[a00]])
            BB = np.array([[b00]])
            one = np.array([[1.0]], dtype=AA.dtype)
            return {"QZ": {"AA": AA, "BB": BB, "Q": one, "Z": one}}

        # Most tasks use real matrices; try float64 first, fallback to complex128.
        try:
            A = np.array(A_in, dtype=np.float64, order="F", copy=True)
            B = np.array(B_in, dtype=np.float64, order="F", copy=True)
            is_real = True
        except (TypeError, ValueError):
            A = np.array(A_in, dtype=np.complex128, order="F", copy=True)
            B = np.array(B_in, dtype=np.complex128, order="F", copy=True)
            is_real = False

        # Prefer direct LAPACK generalized Schur (QZ) decomposition.
        # Returns: A = VSL*S*VSR^H, B = VSL*T*VSR^H
        if is_real and _dgges is not None:
            try:
                res = _dgges(
                    A,
                    B,
                    jobvsl="V",
                    jobvsr="V",
                    sort="N",
                    overwrite_a=True,
                    overwrite_b=True,
                )
                info = res[-1]
                if info == 0:
                    S = res[0]
                    T = res[1]
                    Q = res[5]
                    Z = res[6]
                    return {"QZ": {"AA": S, "BB": T, "Q": Q, "Z": Z}}
            except TypeError:
                # Different SciPy versions have slightly different signatures.
                pass

        if (not is_real) and _zgges is not None:
            try:
                res = _zgges(
                    A,
                    B,
                    jobvsl="V",
                    jobvsr="V",
                    sort="N",
                    overwrite_a=True,
                    overwrite_b=True,
                )
                info = res[-1]
                if info == 0:
                    S = res[0]
                    T = res[1]
                    Q = res[4]
                    Z = res[5]
                    return {"QZ": {"AA": S, "BB": T, "Q": Q, "Z": Z}}
            except TypeError:
                pass

        # Fallback: scipy.linalg.qz (still fast in LAPACK, but more wrapper overhead).
        if _qz is None:  # pragma: no cover
            raise RuntimeError("scipy.linalg.qz is unavailable in this environment")

        AA, BB, Q, Z = _qz(
            A,
            B,
            output="real" if is_real else "complex",
            check_finite=False,
            overwrite_a=True,
            overwrite_b=True,
        )
        return {"QZ": {"AA": AA, "BB": BB, "Q": Q, "Z": Z}}