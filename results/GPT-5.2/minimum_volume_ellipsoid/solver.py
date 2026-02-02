from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

# --------------------------- MVEE (Khachiyan) ---------------------------

if njit is not None:

    @njit(cache=True, fastmath=True)
    def _mvee_khachiyan_numba(P: np.ndarray, tol: float, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Numba Khachiyan with rank-1 inverse updates + periodic recomputation for stability.

        Key optimization: maintain XinvQ = Xinv @ Q and update it with Woodbury too,
        avoiding an O(d^2 n) matmul each iteration.
        """
        n, d = P.shape
        if n == 0:
            return np.eye(d), np.zeros(d)

        dp1 = d + 1

        # Q = [P^T; 1...1]  shape (d+1, n)
        Q = np.empty((dp1, n), dtype=np.float64)
        for i in range(n):
            for k in range(d):
                Q[k, i] = P[i, k]
            Q[d, i] = 1.0

        u = np.full(n, 1.0 / n, dtype=np.float64)

        # Initial X and Xinv
        X = (Q * u) @ Q.T
        for k in range(dp1):
            X[k, k] += 1e-14
        Xinv = np.linalg.inv(X)

        # Maintain XinvQ and update in O(dp1*n)
        XinvQ = Xinv @ Q

        # Workspaces
        v = np.empty(dp1, dtype=np.float64)
        vtQ = np.empty(n, dtype=np.float64)

        for it in range(max_iter):
            # Periodically recompute inverse to reduce drift from Woodbury updates
            if it % 120 == 0 and it > 0:
                for a in range(dp1):
                    for b in range(a + 1, dp1):
                        vv = 0.5 * (X[a, b] + X[b, a])
                        X[a, b] = vv
                        X[b, a] = vv
                Xinv = np.linalg.inv(X)
                XinvQ = Xinv @ Q

            # M_i = q_i^T Xinv q_i = dot(Q[:,i], XinvQ[:,i])
            j = 0
            maxM = -1e300
            for i in range(n):
                s = 0.0
                for k in range(dp1):
                    s += Q[k, i] * XinvQ[k, i]
                if s > maxM:
                    maxM = s
                    j = i

            eps = maxM / dp1 - 1.0
            if eps <= tol:
                break

            step = (maxM - dp1) / (dp1 * (maxM - 1.0))
            if step <= 0.0:
                break

            alpha = 1.0 - step
            beta = step

            # Update u
            u *= alpha
            u[j] += beta

            # v = Xinv q_j is exactly column j of XinvQ (copy before in-place updates)
            for k in range(dp1):
                v[k] = XinvQ[k, j]

            # Update X = alpha X + beta q q^T
            for a in range(dp1):
                qa = Q[a, j]
                for b in range(dp1):
                    X[a, b] = alpha * X[a, b] + beta * qa * Q[b, j]

            # Woodbury update for Xinv and XinvQ
            denom = alpha + beta * maxM
            if denom <= 0.0:
                break
            coeff = beta / denom
            inv_alpha = 1.0 / alpha

            # Xinv update (small dp1^2 cost)
            for a in range(dp1):
                va = v[a]
                for b in range(dp1):
                    Xinv[a, b] = (Xinv[a, b] - coeff * va * v[b]) * inv_alpha

            # vtQ = v^T Q  (length n)
            for i in range(n):
                s = 0.0
                for k in range(dp1):
                    s += v[k] * Q[k, i]
                vtQ[i] = s

            # XinvQ_new = (XinvQ - coeff * v * vtQ) / alpha
            for i in range(n):
                t = coeff * vtQ[i]
                for k in range(dp1):
                    XinvQ[k, i] = (XinvQ[k, i] - v[k] * t) * inv_alpha

        # center c = sum u_i p_i
        c = np.zeros(d, dtype=np.float64)
        for i in range(n):
            ui = u[i]
            for k in range(d):
                c[k] += ui * P[i, k]

        # cov = sum u_i p_i p_i^T - c c^T
        cov = np.zeros((d, d), dtype=np.float64)
        for i in range(n):
            ui = u[i]
            for a in range(d):
                pa = P[i, a]
                for b in range(d):
                    cov[a, b] += ui * pa * P[i, b]

        for a in range(d):
            ca = c[a]
            for b in range(d):
                cov[a, b] -= ca * c[b]

        # Symmetrize + regularize
        for a in range(d):
            for b in range(a + 1, d):
                vv = 0.5 * (cov[a, b] + cov[b, a])
                cov[a, b] = vv
                cov[b, a] = vv
        for k in range(d):
            cov[k, k] += 1e-12

        A = np.linalg.inv(cov) / d
        return A, c

else:
    _mvee_khachiyan_numba = None

def _mvee_khachiyan_numpy(points: np.ndarray, tol: float, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
    """Numpy fallback (stable, slower)."""
    P = np.asarray(points, dtype=np.float64)
    n, d = P.shape
    if n == 0:
        return np.eye(d, dtype=np.float64), np.zeros(d, dtype=np.float64)

    Q = np.empty((d + 1, n), dtype=np.float64)
    Q[:d, :] = P.T
    Q[d, :] = 1.0

    u = np.full(n, 1.0 / n, dtype=np.float64)
    dp1 = float(d + 1)

    X = (Q * u) @ Q.T
    X.flat[:: d + 2] += 1e-14
    Xinv = np.linalg.inv(X)
    XinvQ = Xinv @ Q

    for it in range(max_iter):
        if it % 60 == 0 and it > 0:
            X = (X + X.T) * 0.5
            Xinv = np.linalg.inv(X)
            XinvQ = Xinv @ Q

        M = np.sum(Q * XinvQ, axis=0)
        j = int(np.argmax(M))
        maxM = float(M[j])

        eps = maxM / dp1 - 1.0
        if eps <= tol:
            break

        step = (maxM - dp1) / (dp1 * (maxM - 1.0))
        if step <= 0.0:
            break

        alpha = 1.0 - step
        beta = step
        u *= alpha
        u[j] += beta

        q = Q[:, j]
        v = XinvQ[:, j].copy()
        X = alpha * X + beta * np.outer(q, q)
        denom = alpha + beta * maxM
        coeff = beta / denom
        Xinv = (Xinv - coeff * np.outer(v, v)) / alpha
        XinvQ = (XinvQ - coeff * np.outer(v, v @ Q)) / alpha

    c = P.T @ u
    cov = P.T @ (u[:, None] * P) - np.outer(c, c)
    cov = (cov + cov.T) * 0.5
    cov.flat[:: d + 1] += 1e-12
    A = np.linalg.inv(cov) / float(d)
    return A, c

def _sqrtm_spd(A: np.ndarray) -> np.ndarray:
    """Symmetric square root for SPD/PSD matrix (via eigen decomposition)."""
    A = (A + A.T) * 0.5
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 1e-15)
    X = (V * np.sqrt(w)) @ V.T
    return (X + X.T) * 0.5

class Solver:
    def __init__(self) -> None:
        # Warm up JIT compilation (init time isn't counted).
        if _mvee_khachiyan_numba is not None:
            P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
            _mvee_khachiyan_numba(P, 1e-3, 5)

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        points = np.asarray(problem["points"], dtype=np.float64)
        n, d = points.shape

        # Conservative parameters to stay within 1% objective tolerance robustly.
        tol = 1e-4
        if d >= 15:
            tol = 7e-5
        if d >= 30:
            tol = 5e-5

        max_iter = 500
        if d >= 15:
            max_iter = 800
        if n >= 1200:
            max_iter = 1000

        if _mvee_khachiyan_numba is not None:
            A, c = _mvee_khachiyan_numba(points, tol, max_iter)
        else:
            A, c = _mvee_khachiyan_numpy(points, tol, max_iter)

        X = _sqrtm_spd(A)
        # X is symmetric by construction; compute Y and feasibility in a fast way
        Y = -(X @ c)

        # Enforce feasibility in X,Y form: max ||X a_i + Y|| <= 1
        Z = points @ X  # X symmetric
        Z += Y
        max_norm2 = float(np.max(np.sum(Z * Z, axis=1)))
        if max_norm2 > 1.0:
            scale = (1.0 - 1e-13) / np.sqrt(max_norm2)
            X *= scale
            Y *= scale

        # Objective must match validator: -log(det(X))
        detX = float(np.linalg.det(X))
        if not np.isfinite(detX) or detX <= 0.0:
            X = X + 1e-12 * np.eye(d)
            detX = float(np.linalg.det(X))

        objective_value = float(-np.log(detX))
        return {"objective_value": objective_value, "ellipsoid": {"X": X, "Y": Y}}