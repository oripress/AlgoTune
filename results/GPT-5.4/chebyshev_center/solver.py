from math import comb
from typing import Any

import numpy as np
from numba import njit
from scipy.optimize import linprog

@njit(cache=True)
def _solve_small_system(mat: np.ndarray, rhs: np.ndarray, dim: int) -> tuple[bool, np.ndarray]:
    a = np.empty((5, 5), dtype=np.float64)
    b = np.empty(5, dtype=np.float64)
    x = np.empty(5, dtype=np.float64)

    for i in range(dim):
        b[i] = rhs[i]
        for j in range(dim):
            a[i, j] = mat[i, j]

    for i in range(dim):
        pivot = i
        max_abs = abs(a[i, i])
        for r in range(i + 1, dim):
            v = abs(a[r, i])
            if v > max_abs:
                max_abs = v
                pivot = r
        if max_abs < 1e-12:
            return False, x

        if pivot != i:
            for c in range(i, dim):
                tmp = a[i, c]
                a[i, c] = a[pivot, c]
                a[pivot, c] = tmp
            tmpb = b[i]
            b[i] = b[pivot]
            b[pivot] = tmpb

        piv = a[i, i]
        for r in range(i + 1, dim):
            factor = a[r, i] / piv
            if factor != 0.0:
                a[r, i] = 0.0
                for c in range(i + 1, dim):
                    a[r, c] -= factor * a[i, c]
                b[r] -= factor * b[i]

    for i in range(dim - 1, -1, -1):
        s = b[i]
        for c in range(i + 1, dim):
            s -= a[i, c] * x[c]
        diag = a[i, i]
        if abs(diag) < 1e-12:
            return False, x
        x[i] = s / diag

    return True, x

@njit(cache=True)
def _is_feasible(g: np.ndarray, h: np.ndarray, total: int, z: np.ndarray, dim: int, tol: float) -> bool:
    for i in range(total):
        s = 0.0
        for j in range(dim):
            s += g[i, j] * z[j]
        if s > h[i] + tol:
            return False
    return True

@njit(cache=True)
def _enum_lp3(g: np.ndarray, h: np.ndarray, total: int, tol: float) -> tuple[bool, np.ndarray]:
    best = np.empty(5, dtype=np.float64)
    mat = np.empty((5, 5), dtype=np.float64)
    rhs = np.empty(5, dtype=np.float64)
    found = False
    best_r = -1e300

    for i in range(total - 2):
        rhs[0] = h[i]
        for c in range(3):
            mat[0, c] = g[i, c]
        for j in range(i + 1, total - 1):
            rhs[1] = h[j]
            for c in range(3):
                mat[1, c] = g[j, c]
            for k in range(j + 1, total):
                rhs[2] = h[k]
                for c in range(3):
                    mat[2, c] = g[k, c]
                ok, z = _solve_small_system(mat, rhs, 3)
                if ok:
                    r = z[2]
                    if r > best_r and _is_feasible(g, h, total, z, 3, tol):
                        best_r = r
                        best[0] = z[0]
                        best[1] = z[1]
                        best[2] = z[2]
                        found = True

    return found, best

@njit(cache=True)
def _enum_lp4(g: np.ndarray, h: np.ndarray, total: int, tol: float) -> tuple[bool, np.ndarray]:
    best = np.empty(5, dtype=np.float64)
    mat = np.empty((5, 5), dtype=np.float64)
    rhs = np.empty(5, dtype=np.float64)
    found = False
    best_r = -1e300

    for i in range(total - 3):
        rhs[0] = h[i]
        for c in range(4):
            mat[0, c] = g[i, c]
        for j in range(i + 1, total - 2):
            rhs[1] = h[j]
            for c in range(4):
                mat[1, c] = g[j, c]
            for k in range(j + 1, total - 1):
                rhs[2] = h[k]
                for c in range(4):
                    mat[2, c] = g[k, c]
                for l in range(k + 1, total):
                    rhs[3] = h[l]
                    for c in range(4):
                        mat[3, c] = g[l, c]
                    ok, z = _solve_small_system(mat, rhs, 4)
                    if ok:
                        r = z[3]
                        if r > best_r and _is_feasible(g, h, total, z, 4, tol):
                            best_r = r
                            best[0] = z[0]
                            best[1] = z[1]
                            best[2] = z[2]
                            best[3] = z[3]
                            found = True

    return found, best

@njit(cache=True)
def _enum_lp5(g: np.ndarray, h: np.ndarray, total: int, tol: float) -> tuple[bool, np.ndarray]:
    best = np.empty(5, dtype=np.float64)
    mat = np.empty((5, 5), dtype=np.float64)
    rhs = np.empty(5, dtype=np.float64)
    found = False
    best_r = -1e300

    for i in range(total - 4):
        rhs[0] = h[i]
        for c in range(5):
            mat[0, c] = g[i, c]
        for j in range(i + 1, total - 3):
            rhs[1] = h[j]
            for c in range(5):
                mat[1, c] = g[j, c]
            for k in range(j + 1, total - 2):
                rhs[2] = h[k]
                for c in range(5):
                    mat[2, c] = g[k, c]
                for l in range(k + 1, total - 1):
                    rhs[3] = h[l]
                    for c in range(5):
                        mat[3, c] = g[l, c]
                    for p in range(l + 1, total):
                        rhs[4] = h[p]
                        for c in range(5):
                            mat[4, c] = g[p, c]
                        ok, z = _solve_small_system(mat, rhs, 5)
                        if ok:
                            r = z[4]
                            if r > best_r and _is_feasible(g, h, total, z, 5, tol):
                                best_r = r
                                best[0] = z[0]
                                best[1] = z[1]
                                best[2] = z[2]
                                best[3] = z[3]
                                best[4] = z[4]
                                found = True

    return found, best

_NUMBA_WARM = False

def _warm_numba() -> None:
    global _NUMBA_WARM
    if _NUMBA_WARM:
        return

    g3 = np.array(
        [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    h3 = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    _enum_lp3(g3, h3, 3, 1e-9)

    g4 = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    h4 = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    _enum_lp4(g4, h4, 4, 1e-9)

    g5 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    h5 = np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    _enum_lp5(g5, h5, 5, 1e-9)

    _NUMBA_WARM = True

class Solver:
    _ENUM_MAX_COMB = 250000
    _TOL = 1e-9

    def __init__(self) -> None:
        _warm_numba()

    @staticmethod
    def _solve_1d(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
        coeff = a[:, 0]
        pos = coeff > 0.0
        neg = coeff < 0.0

        if not np.any(pos) or not np.any(neg):
            return None

        upper = np.min(b[pos] / coeff[pos])
        lower = np.max(b[neg] / coeff[neg])
        x = 0.5 * (lower + upper)
        if np.isfinite(x):
            return np.array([x], dtype=np.float64)
        return None

    @classmethod
    def _enumerate_bases(
        cls, a: np.ndarray, b: np.ndarray, norms: np.ndarray
    ) -> np.ndarray | None:
        m, n = a.shape
        dim = n + 1
        total = m + 1

        if dim < 3 or dim > 5 or total < dim:
            return None

        if comb(total, dim) > cls._ENUM_MAX_COMB:
            return None

        g = np.empty((total, dim), dtype=np.float64)
        g[:m, :n] = a / norms[:, None]
        g[:m, n] = 1.0
        g[m, :n] = 0.0
        g[m, n] = -1.0

        h = np.empty(total, dtype=np.float64)
        h[:m] = b / norms
        h[m] = 0.0

        if dim == 3:
            found, z = _enum_lp3(g, h, total, cls._TOL)
        elif dim == 4:
            found, z = _enum_lp4(g, h, total, cls._TOL)
        else:
            found, z = _enum_lp5(g, h, total, cls._TOL)

        if found and np.isfinite(z[n]):
            return z[:n].copy()
        return None

    @staticmethod
    def _fallback_linprog(a: np.ndarray, b: np.ndarray, norms: np.ndarray) -> np.ndarray | None:
        m, n = a.shape

        c = np.zeros(n + 1, dtype=np.float64)
        c[-1] = -1.0

        a_ub = np.empty((m, n + 1), dtype=np.float64)
        a_ub[:, :n] = a
        a_ub[:, n] = norms

        bounds = [(None, None)] * n + [(0.0, None)]

        res = linprog(
            c,
            A_ub=a_ub,
            b_ub=b,
            bounds=bounds,
            method="highs-ds",
            options={"presolve": False},
        )

        if res.success and res.x is not None:
            x = np.asarray(res.x[:n], dtype=np.float64)
            if np.all(np.isfinite(x)):
                return x

        a_ub[:, :n] = a / norms[:, None]
        a_ub[:, n] = 1.0
        res = linprog(
            c,
            A_ub=a_ub,
            b_ub=b / norms,
            bounds=bounds,
            method="highs",
        )

        if res.success and res.x is not None:
            x = np.asarray(res.x[:n], dtype=np.float64)
            if np.all(np.isfinite(x)):
                return x

        return None

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[float]]:
        a = np.asarray(problem["a"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        if a.ndim != 2:
            a = np.atleast_2d(a)

        m, n = a.shape
        if m == 0:
            return {"solution": [0.0] * n}

        norms = np.sqrt(np.einsum("ij,ij->i", a, a))
        nz = norms > 0.0

        if not np.all(nz):
            if np.any(b[~nz] < 0.0):
                return {"solution": [0.0] * n}
            a = a[nz]
            b = b[nz]
            norms = norms[nz]
            m = a.shape[0]
            if m == 0:
                return {"solution": [0.0] * n}

        x = None
        if n == 1:
            x = self._solve_1d(a, b)
        if x is None:
            x = self._enumerate_bases(a, b, norms)
        if x is None:
            x = self._fallback_linprog(a, b, norms)
        if x is None:
            x = np.zeros(n, dtype=np.float64)

        return {"solution": np.asarray(x, dtype=np.float64).tolist()}