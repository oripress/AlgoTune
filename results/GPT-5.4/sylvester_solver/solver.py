from typing import Any

import numpy as np
from scipy.linalg import schur
from scipy.linalg.lapack import cgees, ctrsyl, zgees, ztrsyl

class Solver:
    def __init__(self) -> None:
        self._trsyl_map = {
            np.dtype(np.complex64): ctrsyl,
            np.dtype(np.complex128): ztrsyl,
        }
        self._gees_map = {
            np.dtype(np.complex64): cgees,
            np.dtype(np.complex128): zgees,
        }

    @staticmethod
    def _select(_: complex) -> int:
        return 0

    def solve(self, problem, **kwargs) -> Any:
        if not isinstance(problem, dict):
            import time

            def gen(n, m, seed=0):
                rng = np.random.default_rng(seed)
                A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
                B = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
                A += (5.0 + 0.1j) * np.eye(n)
                B += (-5.0 + 0.2j) * np.eye(m)
                Q = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
                return A, B, Q

            def schur_fast(A, B, Q):
                r, u = schur(A, output="complex", check_finite=False)
                s, v = schur(B, output="complex", check_finite=False)
                y, scale, info = ztrsyl(r, s, u.conj().T @ Q @ v)
                if scale != 1:
                    y = y * scale
                return u @ y @ v.conj().T

            def gees_fast(A, B, Q):
                A1 = np.array(A, dtype=np.complex128, order="F", copy=True)
                B1 = np.array(B, dtype=np.complex128, order="F", copy=True)
                r, _, _, u, _, info = zgees(self._select, A1, compute_v=1, sort_t=0, overwrite_a=1)
                s, _, _, v, _, info2 = zgees(self._select, B1, compute_v=1, sort_t=0, overwrite_a=1)
                y, scale, info3 = ztrsyl(r, s, u.conj().T @ Q @ v)
                if scale != 1:
                    y = y * scale
                return u @ y @ v.conj().T

            for n, m in [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]:
                A, B, Q = gen(n, m, 0)
                x0 = schur_fast(A, B, Q)
                for fn, name in ((schur_fast, "schur"), (gees_fast, "gees")):
                    x1 = fn(A, B, Q)
                    print("shape", n, m, name, "res", np.linalg.norm(A @ x1 + x1 @ B - Q), "err", np.linalg.norm(x1 - x0))
                    t0 = time.perf_counter()
                    for _ in range(20):
                        fn(A, B, Q)
                    print(name, time.perf_counter() - t0)
            return {"X": np.zeros((1, 1), dtype=np.complex128)}

        A = problem["A"]
        B = problem["B"]
        Q = problem["Q"]

        r, u = schur(A, output="complex", check_finite=False)
        s, v = schur(B, output="complex", check_finite=False)
        f = u.conj().T @ Q @ v

        trsyl = self._trsyl_map.get(r.dtype, ztrsyl)
        y, scale, info = trsyl(r, s, f)
        if info < 0:
            raise ValueError(f"Invalid trsyl argument: {info}")
        if scale != 1:
            y = y * scale

        return {"X": u @ y @ v.conj().T}