from typing import Any

import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _diag_qhq(q: np.ndarray, hq: np.ndarray) -> np.ndarray:
    m, n = q.shape
    out = np.empty(n, dtype=np.float64)
    for j in range(n):
        s = 0.0
        for i in range(m):
            s += q[i, j] * hq[i, j]
        out[j] = s
    return out

@njit(cache=True, fastmath=True)
def _build_q(points: np.ndarray) -> np.ndarray:
    n, d = points.shape
    q = np.empty((d + 1, n), dtype=np.float64)
    for i in range(n):
        for k in range(d):
            q[k, i] = points[i, k]
        q[d, i] = 1.0
    return q

@njit(cache=True, fastmath=True)
def _safe_max_norm(points: np.ndarray, xmat: np.ndarray, yvec: np.ndarray) -> float:
    n, d = points.shape
    best = 0.0
    for i in range(n):
        s = 0.0
        for a in range(d):
            t = yvec[a]
            for b in range(d):
                t += xmat[a, b] * points[i, b]
            s += t * t
        val = np.sqrt(s)
        if val > best:
            best = val
    return best

@njit(cache=True, fastmath=True)
def _mvee_core(
    points: np.ndarray,
    tol: float,
    max_iter: int,
    refresh: int,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    n, d = points.shape
    m = d + 1

    q = _build_q(points)
    u = np.full(n, 1.0 / n, dtype=np.float64)
    xmat = np.dot(q * u, q.T)
    hmat = np.linalg.inv(xmat)
    hq = np.dot(hmat, q)
    mvals = _diag_qhq(q, hq)
    threshold = m * (1.0 + tol)

    for it in range(max_iter):
        j = int(np.argmax(mvals))
        max_m = mvals[j]
        if not np.isfinite(max_m):
            return 0, 0.0, np.empty((d, d), dtype=np.float64), np.empty(d, dtype=np.float64)
        if max_m <= threshold:
            break

        alpha = (max_m - m) / (m * (max_m - 1.0))
        if not np.isfinite(alpha) or alpha <= 0.0:
            break
        beta = 1.0 - alpha
        if beta <= 0.0:
            return 0, 0.0, np.empty((d, d), dtype=np.float64), np.empty(d, dtype=np.float64)

        qj = q[:, j]
        hqj = np.dot(hmat, qj)
        z = np.dot(q.T, hqj)
        denom = beta + alpha * max_m
        if denom <= 0.0 or not np.isfinite(denom):
            return 0, 0.0, np.empty((d, d), dtype=np.float64), np.empty(d, dtype=np.float64)
        coef = alpha / denom

        hmat = (hmat - coef * np.outer(hqj, hqj)) / beta
        mvals = (mvals - coef * z * z) / beta
        u *= beta
        u[j] += alpha

        if refresh > 0 and (it + 1) % refresh == 0:
            xmat = np.dot(q * u, q.T)
            hmat = np.linalg.inv(xmat)
            hq = np.dot(hmat, q)
            mvals = _diag_qhq(q, hq)

    xmat = np.dot(q * u, q.T)
    hmat = np.linalg.inv(xmat)
    hq = np.dot(hmat, q)
    mvals = _diag_qhq(q, hq)
    if np.max(mvals) > m * 1.01:
        return 0, 0.0, np.empty((d, d), dtype=np.float64), np.empty(d, dtype=np.float64)

    center = np.dot(u, points)
    diff = points - center
    scat = np.dot((diff * u.reshape(n, 1)).T, diff)

    evals, evecs = np.linalg.eigh(scat)
    max_eval = np.max(evals)
    if not np.isfinite(max_eval) or max_eval <= 0.0:
        return 0, 0.0, np.empty((d, d), dtype=np.float64), np.empty(d, dtype=np.float64)

    floor = max(1e-15, 1e-14 * max_eval)
    if np.min(evals) <= floor:
        return 0, 0.0, np.empty((d, d), dtype=np.float64), np.empty(d, dtype=np.float64)

    sqrt_inv = 1.0 / np.sqrt(d * evals)
    xmat_out = np.dot(evecs * sqrt_inv, evecs.T)
    yvec = -np.dot(xmat_out, center)

    max_norm = _safe_max_norm(points, xmat_out, yvec)
    scale = 1.0
    if not np.isfinite(max_norm):
        return 0, 0.0, np.empty((d, d), dtype=np.float64), np.empty(d, dtype=np.float64)
    if max_norm > 1.0:
        scale = max_norm * (1.0 + 1e-12)
        xmat_out /= scale
        yvec /= scale

    obj = 0.0
    for i in range(d):
        obj += 0.5 * np.log(d * evals[i])
    if scale > 1.0:
        obj += d * np.log(scale)

    return 1, obj, xmat_out, yvec

class Solver:
    def __init__(self) -> None:
        self._cp = None
        self._jit_ready = False
        try:
            dummy = np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=np.float64,
            )
            _mvee_core(dummy, 1e-5, 4, 2)
            self._jit_ready = True
        except Exception:
            self._jit_ready = False

    def _nan_result(self, d: int) -> dict[str, Any]:
        return {
            "objective_value": float("inf"),
            "ellipsoid": {
                "X": np.full((d, d), np.nan, dtype=float),
                "Y": np.full(d, np.nan, dtype=float),
            },
        }

    def _fallback(self, points: np.ndarray) -> dict[str, Any]:
        try:
            if self._cp is None:
                import cvxpy as cp

                self._cp = cp
            cp = self._cp
            n, d = points.shape
            X = cp.Variable((d, d), symmetric=True)
            Y = cp.Variable((d,))
            constraints = [cp.SOC(1, X @ points[i] + Y) for i in range(n)]
            prob = cp.Problem(cp.Minimize(-cp.log_det(X)), constraints)
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if prob.status not in {"optimal", "optimal_inaccurate"}:
                return self._nan_result(d)
            return {
                "objective_value": float(prob.value),
                "ellipsoid": {"X": np.asarray(X.value), "Y": np.asarray(Y.value)},
            }
        except Exception:
            return self._nan_result(points.shape[1])

    def solve(self, problem, **kwargs) -> Any:
        points = np.asarray(problem["points"], dtype=np.float64)
        if points.ndim != 2:
            return self._nan_result(0)

        n, d = points.shape
        if n == 0 or d == 0:
            return self._nan_result(d)

        if d == 1 and n >= 2:
            lo = float(np.min(points[:, 0]))
            hi = float(np.max(points[:, 0]))
            width = hi - lo
            if width > 0.0 and np.isfinite(width):
                x = np.array([[2.0 / width]], dtype=np.float64)
                y = np.array([-(hi + lo) / width], dtype=np.float64)
                return {
                    "objective_value": float(-np.log(x[0, 0])),
                    "ellipsoid": {"X": x, "Y": y},
                }
        if n > 512:
            try:
                points = np.unique(points, axis=0)
                n = points.shape[0]
            except Exception:
                pass

        if d <= 6 and n >= max(32, 4 * (d + 1)):
            try:
                hull = ConvexHull(points)
                verts = hull.vertices
                if verts.size > d and verts.size < n:
                    points = points[verts]
                    n = points.shape[0]
            except Exception:
                pass

        if n <= d:
            return self._fallback(points)
            return self._fallback(points)

        tol = float(kwargs.get("tol", 3e-5))
        tol = float(kwargs.get("tol", 1e-3))
        max_iter = int(kwargs.get("max_iter", 600 + 40 * d))
        refresh = int(kwargs.get("refresh", 0))
        try:
            ok, obj, xmat, yvec = _mvee_core(points, tol, max_iter, refresh)
        except Exception:
            ok = 0
            obj = 0.0
            xmat = np.empty((d, d), dtype=np.float64)
            yvec = np.empty(d, dtype=np.float64)

        if ok != 1 or not np.isfinite(obj):
            return self._fallback(points)

        return {
            "objective_value": float(obj),
            "ellipsoid": {"X": xmat, "Y": yvec},
        }