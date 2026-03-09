from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

class Solver:
    __slots__ = ("_eye_cache", "_p1", "_p2")

    def __init__(self) -> None:
        self._eye_cache: dict[int, np.ndarray] = {}
        self._p1 = ((1.0,),)
        self._p2 = ((1.0, 0.0), (0.0, 1.0))

    def _eye(self, n: int) -> np.ndarray:
        eye = self._eye_cache.get(n)
        if eye is None:
            eye = np.eye(n, dtype=float)
            self._eye_cache[n] = eye
        return eye

    def solve(self, problem, **kwargs) -> Any:
        try:
            A = np.asarray(problem["A"], dtype=float)
            if A.ndim != 2:
                return {"is_stable": False, "P": None}

            n, m = A.shape
            if n != m or n == 0:
                return {"is_stable": False, "P": None}

            if n == 1:
                a = float(A[0, 0])
                if abs(a) >= 1.0 or not np.isfinite(a):
                    return {"is_stable": False, "P": None}
                return {"is_stable": True, "P": self._p1}

            if n == 2:
                a = float(A[0, 0])
                b = float(A[0, 1])
                c = float(A[1, 0])
                d = float(A[1, 1])

                s00 = 1.0 - a * a - c * c
                s01 = -(a * b + c * d)
                s11 = 1.0 - b * b - d * d
                if s00 > 1e-12 and (s00 * s11 - s01 * s01) > 1e-12:
                    return {"is_stable": True, "P": self._p2}

                a2 = a * a + b * c
                b2 = b * (a + d)
                c2 = c * (a + d)
                d2 = d * d + b * c
                t00 = 1.0 - a2 * a2 - c2 * c2
                t01 = -(a2 * b2 + c2 * d2)
                t11 = 1.0 - b2 * b2 - d2 * d2
                if t00 > 1e-12 and (t00 * t11 - t01 * t01) > 1e-12:
                    p01 = a * b + c * d
                    return {
                        "is_stable": True,
                        "P": (
                            (1.0 + a * a + c * c, p01),
                            (p01, 1.0 + b * b + d * d),
                        ),
                    }

                det = a * d - b * c
                tr = a + d
                if (1.0 - tr + det) <= 0.0 or (1.0 + tr + det) <= 0.0 or (1.0 - det) <= 0.0:
                    return {"is_stable": False, "P": None}

                M = np.array(
                    [
                        [1.0 - a * a, -2.0 * a * c, -c * c],
                        [-a * b, 1.0 - a * d - b * c, -c * d],
                        [-b * b, -2.0 * b * d, 1.0 - d * d],
                    ],
                    dtype=float,
                )
                rhs = np.array([1.0, 0.0, 1.0], dtype=float)
                x = np.linalg.solve(M, rhs)
                if x[0] <= 1e-12 or (x[0] * x[2] - x[1] * x[1]) <= 1e-12:
                    return {"is_stable": False, "P": None}
                return {"is_stable": True, "P": ((x[0], x[1]), (x[1], x[2]))}

            eye = self._eye(n)
            P = eye
            B = A
            for _ in range(4):
                G = B.T @ B
                try:
                    np.linalg.cholesky(eye - G)
                    return {"is_stable": True, "P": P}
                except Exception:
                    P = P + G
                    B = B @ A

            if np.max(np.abs(np.linalg.eigvals(A))) >= 1.0:
                return {"is_stable": False, "P": None}

            P = solve_discrete_lyapunov(A.T, eye)
            P = 0.5 * (P + P.T)
            if not np.isfinite(P).all():
                return {"is_stable": False, "P": None}
            np.linalg.cholesky(P)
            return {"is_stable": True, "P": P}
        except Exception:
            return {"is_stable": False, "P": None}