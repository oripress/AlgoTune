from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# HiGHS-backed LP (fast, compiled). Much lower overhead than CVXPY for this task.
try:
    from scipy.optimize import linprog  # type: ignore
except Exception:  # pragma: no cover
    linprog = None  # type: ignore

try:
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover
    njit = None  # type: ignore

def _comb_int(n: int, k: int) -> int:
    # Small helper to avoid importing math.comb (tiny overhead), and n,k are small.
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    c = 1
    for i in range(1, k + 1):
        c = (c * (n - k + i)) // i
    return c

if njit is not None:

    @njit(cache=True, fastmath=True)
    def _solve3(a0, a1, a2, b0, b1, b2):
        # Gaussian elimination with partial pivoting for 3x3.
        M = np.empty((3, 3), dtype=np.float64)
        rhs = np.empty(3, dtype=np.float64)
        M[0, 0], M[0, 1], M[0, 2] = a0[0], a0[1], a0[2]
        M[1, 0], M[1, 1], M[1, 2] = a1[0], a1[1], a1[2]
        M[2, 0], M[2, 1], M[2, 2] = a2[0], a2[1], a2[2]
        rhs[0], rhs[1], rhs[2] = b0, b1, b2

        for col in range(3):
            piv = col
            maxv = abs(M[col, col])
            for r in range(col + 1, 3):
                v = abs(M[r, col])
                if v > maxv:
                    maxv = v
                    piv = r
            if maxv < 1e-14:
                return np.zeros(3, dtype=np.float64), False
            if piv != col:
                for cc in range(col, 3):
                    tmp = M[col, cc]
                    M[col, cc] = M[piv, cc]
                    M[piv, cc] = tmp
                tmp = rhs[col]
                rhs[col] = rhs[piv]
                rhs[piv] = tmp

            pivv = M[col, col]
            invp = 1.0 / pivv
            for cc in range(col, 3):
                M[col, cc] *= invp
            rhs[col] *= invp

            for r in range(3):
                if r == col:
                    continue
                f = M[r, col]
                if f != 0.0:
                    for cc in range(col, 3):
                        M[r, cc] -= f * M[col, cc]
                    rhs[r] -= f * rhs[col]

        return rhs, True

    @njit(cache=True, fastmath=True)
    def _solve4(a0, a1, a2, a3, b0, b1, b2, b3):
        # Gaussian elimination with partial pivoting for 4x4.
        M = np.empty((4, 4), dtype=np.float64)
        rhs = np.empty(4, dtype=np.float64)
        for j in range(4):
            M[0, j] = a0[j]
            M[1, j] = a1[j]
            M[2, j] = a2[j]
            M[3, j] = a3[j]
        rhs[0], rhs[1], rhs[2], rhs[3] = b0, b1, b2, b3

        for col in range(4):
            piv = col
            maxv = abs(M[col, col])
            for r in range(col + 1, 4):
                v = abs(M[r, col])
                if v > maxv:
                    maxv = v
                    piv = r
            if maxv < 1e-14:
                return np.zeros(4, dtype=np.float64), False
            if piv != col:
                for cc in range(col, 4):
                    tmp = M[col, cc]
                    M[col, cc] = M[piv, cc]
                    M[piv, cc] = tmp
                tmp = rhs[col]
                rhs[col] = rhs[piv]
                rhs[piv] = tmp

            pivv = M[col, col]
            invp = 1.0 / pivv
            for cc in range(col, 4):
                M[col, cc] *= invp
            rhs[col] *= invp

            for r in range(4):
                if r == col:
                    continue
                f = M[r, col]
                if f != 0.0:
                    for cc in range(col, 4):
                        M[r, cc] -= f * M[col, cc]
                    rhs[r] -= f * rhs[col]

        return rhs, True

    @njit(cache=True, fastmath=True)
    def _enum_vertices(A: np.ndarray, b: np.ndarray, d: int) -> np.ndarray:
        """
        Enumerate all d-constraint vertices (d=3 or 4) and pick the feasible one
        with maximum r (last coordinate).
        A: (N,d) inequalities A y <= b, includes extra constraint -r <= 0.
        """
        N = A.shape[0]
        best = np.zeros(d, dtype=np.float64)
        best_r = -1.0e300
        tol = 1e-9
        # slightly looser feasibility tolerance to avoid rejecting numerically valid points
        ftol = 1e-7

        if d == 3:
            for i in range(N - 2):
                ai = A[i]
                bi = b[i]
                for j in range(i + 1, N - 1):
                    aj = A[j]
                    bj = b[j]
                    for k in range(j + 1, N):
                        ak = A[k]
                        bk = b[k]
                        y, ok = _solve3(ai, aj, ak, bi, bj, bk)
                        if not ok:
                            continue
                        r = y[2]
                        if r < -tol or r <= best_r:
                            continue
                        # feasibility check
                        x0 = y[0]
                        x1 = y[1]
                        feas = True
                        for t in range(N):
                            at = A[t]
                            if at[0] * x0 + at[1] * x1 + at[2] * r - b[t] > ftol:
                                feas = False
                                break
                        if feas:
                            best_r = r
                            best[0] = x0
                            best[1] = x1
                            best[2] = r

        else:  # d == 4
            for i in range(N - 3):
                ai = A[i]
                bi = b[i]
                for j in range(i + 1, N - 2):
                    aj = A[j]
                    bj = b[j]
                    for k in range(j + 1, N - 1):
                        ak = A[k]
                        bk = b[k]
                        for l in range(k + 1, N):
                            al = A[l]
                            bl = b[l]
                            y, ok = _solve4(ai, aj, ak, al, bi, bj, bk, bl)
                            if not ok:
                                continue
                            r = y[3]
                            if r < -tol or r <= best_r:
                                continue
                            x0 = y[0]
                            x1 = y[1]
                            x2 = y[2]
                            feas = True
                            for t in range(N):
                                at = A[t]
                                if (
                                    at[0] * x0
                                    + at[1] * x1
                                    + at[2] * x2
                                    + at[3] * r
                                    - b[t]
                                    > ftol
                                ):
                                    feas = False
                                    break
                            if feas:
                                best_r = r
                                best[0] = x0
                                best[1] = x1
                                best[2] = x2
                                best[3] = r

        return best

class Solver:
    """
    Chebyshev center via an LP.

    Fast path: for small dimensions (n<=3), enumerate vertices in (x,r) space
    using a Numba-compiled kernel (avoids ~ms-level external solver overhead).

    Fallback: HiGHS via SciPy linprog.
    """

    def __init__(self) -> None:
        self._cvxpy: Optional[Any] = None

        # Trigger numba compilation in init (not counted in runtime).
        if njit is not None:
            A3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
            b3 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            _ = _enum_vertices(A3, b3, 3)
            A4 = np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, -1.0]],
                dtype=np.float64,
            )
            b4 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            _ = _enum_vertices(A4, b4, 4)

    @staticmethod
    def _as_float_array(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    def _solve_with_highs(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if linprog is None:
            raise RuntimeError("scipy.optimize.linprog unavailable")

        m, n = a.shape
        norms = np.sqrt(np.einsum("ij,ij->i", a, a))

        A_ub = np.empty((m, n + 1), dtype=np.float64)
        A_ub[:, :n] = a
        A_ub[:, n] = norms

        c = np.zeros(n + 1, dtype=np.float64)
        c[n] = -1.0  # minimize -r

        bounds = [(None, None)] * n + [(0.0, None)]

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b,
            bounds=bounds,
            method="highs-ds",
            options={"presolve": False, "disp": False},
        )
        if not res.success or res.x is None:
            res = linprog(
                c,
                A_ub=A_ub,
                b_ub=b,
                bounds=bounds,
                method="highs",
                options={"presolve": True, "disp": False},
            )
            if not res.success or res.x is None:
                raise RuntimeError(f"HiGHS failed: {res.message}")
        return res.x[:n]

    def _solve_with_cvxpy(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self._cvxpy is None:
            import cvxpy as cp  # type: ignore

            self._cvxpy = cp
        cp = self._cvxpy

        n = a.shape[1]
        x = cp.Variable(n)
        r = cp.Variable()
        norms = np.linalg.norm(a, axis=1)
        prob = cp.Problem(cp.Maximize(r), [a @ x + r * norms <= b, r >= 0])
        prob.solve(solver="CLARABEL")
        if prob.status != "optimal":
            raise RuntimeError(f"CVXPY failed with status={prob.status}")
        return np.asarray(x.value, dtype=np.float64)

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, List[float]]:
        a = self._as_float_array(problem["a"])
        b = self._as_float_array(problem["b"]).reshape(-1)
        m, n = a.shape

        # Very fast small-dimension path (avoid external solver overhead).
        # Enumerate vertices in d=n+1 variables y=[x,r].
        if njit is not None and n <= 3:
            d = n + 1
            N = m + 1  # add -r <= 0
            combos = _comb_int(N, d)
            # Only use enumeration if it's not going to blow up.
            if (d == 3 and combos <= 400_000) or (d == 4 and combos <= 1_200_000):
                norms = np.sqrt(np.einsum("ij,ij->i", a, a))
                Aall = np.empty((N, d), dtype=np.float64)
                Aall[:m, :n] = a
                Aall[:m, n] = norms
                Aall[m, :n] = 0.0
                Aall[m, n] = -1.0
                ball = np.empty(N, dtype=np.float64)
                ball[:m] = b
                ball[m] = 0.0

                y = _enum_vertices(Aall, ball, d)
                x = y[:n]
                if np.all(np.isfinite(x)):
                    return {"solution": x.tolist()}
                # fall through to HiGHS fallback on numerical oddities

        try:
            x = self._solve_with_highs(a, b)
        except Exception:
            x = self._solve_with_cvxpy(a, b)

        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return {"solution": x.tolist()}