from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    """
    Fast matrix square root.

    - n=0/1: trivial
    - n=2: closed-form 2x2
    - n<=eig_max_n: eigendecomposition + right-side solve (often faster for small n)
    - fallback/large: scipy.linalg.sqrtm with disp=True and check_finite=False (if supported)
    """

    def __init__(self) -> None:
        import inspect
        import scipy.linalg  # type: ignore

        self._sqrtm = scipy.linalg.sqrtm
        sig = inspect.signature(self._sqrtm)
        params = sig.parameters
        self._sqrtm_kwargs: dict[str, Any] = {"disp": True}
        if "check_finite" in params:
            self._sqrtm_kwargs["check_finite"] = False

        self._eig_max_n = 32

    @staticmethod
    def _sqrt2x2(A: np.ndarray) -> np.ndarray | None:
        a = complex(A[0, 0])
        b = complex(A[0, 1])
        c = complex(A[1, 0])
        d = complex(A[1, 1])

        tr = a + d
        det = a * d - b * c
        s0 = complex(np.sqrt(det))

        best: tuple[complex, complex, complex, complex] | None = None
        best_err = float("inf")
        for s in (s0, -s0):
            u = complex(np.sqrt(tr + 2.0 * s))
            if abs(u) < 1e-14:
                continue
            invu = 1.0 / u
            x11 = (a + s) * invu
            x12 = b * invu
            x21 = c * invu
            x22 = (d + s) * invu

            r11 = (x11 * x11 + x12 * x21) - a
            r12 = (x11 * x12 + x12 * x22) - b
            r21 = (x21 * x11 + x22 * x21) - c
            r22 = (x21 * x12 + x22 * x22) - d
            err = max(abs(r11), abs(r12), abs(r21), abs(r22))
            if err < best_err and np.isfinite(err):
                best_err = err
                best = (x11, x12, x21, x22)

        if best is None:
            return None
        x11, x12, x21, x22 = best
        return np.array([[x11, x12], [x21, x22]], dtype=np.complex128)

    @staticmethod
    def _eig_sqrt(A: np.ndarray) -> np.ndarray:
        w, V = np.linalg.eig(A)
        sw = np.sqrt(w)
        B = (V * sw).T
        return np.linalg.solve(V.T, B).T

    def solve(self, problem: dict, **kwargs) -> Any:
        if isinstance(problem, str):
            import ast

            problem = ast.literal_eval(problem)

        A_in = problem["matrix"]
        if isinstance(A_in, np.ndarray):
            A = A_in
        else:
            A = np.asarray(A_in)

        n = A.shape[0]
        if n == 0:
            return {"sqrtm": {"X": []}}
        if n == 1:
            return {"sqrtm": {"X": [[complex(np.sqrt(A[0, 0]))]]}}

        if n == 2:
            X2 = self._sqrt2x2(A)
            if X2 is not None and np.all(np.isfinite(X2)):
                return {"sqrtm": {"X": X2.tolist()}}

        if n <= self._eig_max_n:
            try:
                X = self._eig_sqrt(A.astype(np.complex128, copy=False))
                if np.all(np.isfinite(X)):
                    return {"sqrtm": {"X": X.tolist()}}
            except Exception:
                pass

        try:
            X = self._sqrtm(A.astype(np.complex128, copy=False), **self._sqrtm_kwargs)
            return {"sqrtm": {"X": X.tolist()}}
        except Exception:
            return {"sqrtm": {"X": []}}