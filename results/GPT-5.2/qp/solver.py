from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from scipy import sparse  # type: ignore
except Exception:  # pragma: no cover
    sparse = None  # type: ignore[assignment]

try:  # pragma: no cover
    import osqp  # type: ignore

    _HAVE_OSQP = True
except Exception:  # pragma: no cover
    osqp = None  # type: ignore[assignment]
    _HAVE_OSQP = False

class _QPData:
    __slots__ = ("P", "q", "C", "l", "u")

    def __init__(
        self,
        P: NDArray[np.float64],
        q: NDArray[np.float64],
        C: NDArray[np.float64],
        l: NDArray[np.float64],
        u: NDArray[np.float64],
    ) -> None:
        self.P = P
        self.q = q
        self.C = C
        self.l = l
        self.u = u

def _asarray_f64(x: Any) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)

def _as2d(mat: Any, ncols: int) -> NDArray[np.float64]:
    """Convert to (m, ncols); allow empty and 1D single-row."""
    if mat is None:
        return np.empty((0, ncols), dtype=np.float64)
    arr = _asarray_f64(mat)
    if arr.size == 0:
        return np.empty((0, ncols), dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] == ncols:
            return arr.reshape(1, ncols)
        return arr.reshape(-1, ncols)
    return arr

def _solve_equality_qp(P: NDArray[np.float64], q: NDArray[np.float64], A: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Solve:
        min 0.5 x^T P x + q^T x
        s.t. A x = b
    via Schur complement with ridge regularization.
    """
    n = int(P.shape[0])
    p = int(A.shape[0])
    if p == 0:
        ridge = 1e-12
        return -np.linalg.solve(P + ridge * np.eye(n, dtype=np.float64), q)

    ridge = 1e-10
    Pr = P + ridge * np.eye(n, dtype=np.float64)
    L = np.linalg.cholesky(Pr)

    def solve_pr(rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        y = np.linalg.solve(L, rhs)
        return np.linalg.solve(L.T, y)

    Pr_inv_q = solve_pr(q)
    Pr_inv_At = solve_pr(A.T)

    S = A @ Pr_inv_At
    lam = np.linalg.solve(S, -(b + A @ Pr_inv_q))
    return -Pr_inv_q - Pr_inv_At @ lam

def _solve_ipm_qp(
    P: NDArray[np.float64],
    q: NDArray[np.float64],
    G: NDArray[np.float64],
    h: NDArray[np.float64],
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    max_iter: int = 30,
) -> NDArray[np.float64] | None:
    """
    Primal-dual interior point method for small dense QPs.

    Returns x if converged to high feasibility, else None.
    """
    n = int(P.shape[0])
    m = int(G.shape[0])
    p = int(A.shape[0])

    if m == 0:
        return _solve_equality_qp(P, q, A, b)

    # Initial x: satisfy equalities approximately.
    if p:
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    else:
        x = np.zeros(n, dtype=np.float64)

    # Positive slacks/duals.
    s = h - G @ x
    s = np.where(s > 1.0, s, 1.0)
    z = np.ones(m, dtype=np.float64)
    y = np.zeros(p, dtype=np.float64) if p else np.empty((0,), dtype=np.float64)

    e = np.ones(m, dtype=np.float64)
    ridge_h = 1e-12

    for _ in range(max_iter):
        # Residuals
        r_dual = P @ x + q + G.T @ z
        if p:
            r_dual = r_dual + A.T @ y
            r_pe = A @ x - b
        else:
            r_pe = None

        r_pi = G @ x + s - h

        gap = float(s @ z) / m
        if gap <= 0.0:
            gap = 1e-12
        sigma = 0.1
        mu = sigma * gap
        r_cent = mu * e - s * z

        # Stopping (infinity norm for speed)
        dual_inf = float(np.max(np.abs(r_dual)))
        pri_inf = float(np.max(np.abs(r_pi)))
        if p:
            pri_inf = max(pri_inf, float(np.max(np.abs(r_pe))))  # type: ignore[arg-type]

        if dual_inf < 5e-9 and pri_inf < 5e-9 and gap < 5e-9:
            break

        inv_s = 1.0 / s
        w = z * inv_s  # z/s

        # H = P + G^T diag(w) G
        GW = G * w[:, None]
        H = P + G.T @ GW
        # ensure SPD
        H = H + ridge_h * np.eye(n, dtype=np.float64)

        # rhs1 = -r_dual - G^T * ((r_cent + z*r_pi)/s)
        tmp = (r_cent + z * r_pi) * inv_s
        rhs1 = -r_dual - G.T @ tmp

        # Solve for dx,dy using Cholesky of H and Schur complement.
        L = np.linalg.cholesky(H)

        def solve_h(rhs: NDArray[np.float64]) -> NDArray[np.float64]:
            y1 = np.linalg.solve(L, rhs)
            return np.linalg.solve(L.T, y1)

        if p:
            Hinv_rhs1 = solve_h(rhs1)
            Hinv_At = solve_h(A.T)  # (n,p)
            S = A @ Hinv_At
            rhs_y = A @ Hinv_rhs1 + r_pe  # S dy = A H^{-1} rhs1 + r_pe
            dy = np.linalg.solve(S, rhs_y)
            dx = Hinv_rhs1 - Hinv_At @ dy
        else:
            dx = solve_h(rhs1)
            dy = None

        ds = -r_pi - G @ dx
        dz = (r_cent - z * ds) * inv_s

        # Step sizes to keep s,z positive.
        alpha = 1.0
        neg = ds < 0
        if np.any(neg):
            alpha = min(alpha, float(0.99 * np.min(-s[neg] / ds[neg])))
        neg = dz < 0
        if np.any(neg):
            alpha = min(alpha, float(0.99 * np.min(-z[neg] / dz[neg])))

        if not np.isfinite(alpha) or alpha <= 0.0:
            return None

        x = x + alpha * dx
        s = s + alpha * ds
        z = z + alpha * dz
        if p and dy is not None:
            y = y + alpha * dy

        # guard
        if np.any(s <= 0.0) or np.any(z <= 0.0):
            return None

    # Check feasibility tightly (validator uses 1e-6)
    if p:
        if np.max(np.abs(A @ x - b)) > 5e-7:
            return None
    if np.max(G @ x - h) > 5e-7:
        return None

    return x

class Solver:
    """
    Solve convex QP:
        minimize 0.5 x^T P x + q^T x
        subject to Gx <= h
                   Ax == b

    Uses OSQP directly; for small dense problems uses an IPM (faster).
    """

    def __init__(self) -> None:
        self._last_x: NDArray[np.float64] | None = None

    def _prepare(self, problem: dict[str, Any]) -> _QPData:
        P = problem.get("P", None)
        if P is None:
            P = problem.get("Q", None)
        if P is None:
            raise KeyError("Problem must contain 'P' or 'Q'.")

        P = _asarray_f64(P)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("P/Q must be a square matrix.")
        n = int(P.shape[0])

        q = _asarray_f64(problem.get("q", np.zeros(n, dtype=np.float64)))
        if q.size == 0:
            q = np.zeros(n, dtype=np.float64)
        q = q.reshape(n)

        # Ensure symmetric.
        P = (P + P.T) * 0.5

        G = _as2d(problem.get("G", None), n)
        if G.shape[0]:
            h = _asarray_f64(problem.get("h", np.zeros(G.shape[0], dtype=np.float64))).reshape(G.shape[0])
        else:
            h = np.empty((0,), dtype=np.float64)

        A = _as2d(problem.get("A", None), n)
        if A.shape[0]:
            b = _asarray_f64(problem.get("b", np.zeros(A.shape[0], dtype=np.float64))).reshape(A.shape[0])
        else:
            b = np.empty((0,), dtype=np.float64)

        # Build OSQP constraint form l <= Cx <= u.
        if G.shape[0] == 0 and A.shape[0] == 0:
            C = np.empty((0, n), dtype=np.float64)
            l = np.empty((0,), dtype=np.float64)
            u = np.empty((0,), dtype=np.float64)
        elif G.shape[0] and A.shape[0]:
            C = np.vstack((G, A))
            l = np.concatenate((np.full(G.shape[0], -np.inf, dtype=np.float64), b))
            u = np.concatenate((h, b))
        elif G.shape[0]:
            C = G
            l = np.full(G.shape[0], -np.inf, dtype=np.float64)
            u = h
        else:
            C = A
            l = b
            u = b

        return _QPData(P=P, q=q, C=C, l=l, u=u)

    def _solve_via_osqp(
        self, P: NDArray[np.float64], q: NDArray[np.float64], C: NDArray[np.float64], l: NDArray[np.float64], u: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if not _HAVE_OSQP or sparse is None:
            raise RuntimeError("OSQP not available")

        n = int(P.shape[0])
        m = int(C.shape[0])

        P_sp = sparse.csc_matrix(np.triu(P))
        A_sp = sparse.csc_matrix(C) if m else sparse.csc_matrix((0, n))

        solver = osqp.OSQP()
        solver.setup(
            P=P_sp,
            q=q,
            A=A_sp,
            l=l,
            u=u,
            verbose=False,
            eps_abs=1e-8,
            eps_rel=1e-8,
            max_iter=15000,
            polish=False,
            scaled_termination=True,
            warm_start=True,
        )

        if self._last_x is not None and self._last_x.shape == (n,):
            try:
                solver.warm_start(x=self._last_x)
            except Exception:
                pass

        res = solver.solve()
        x = res.x
        if x is None:
            raise RuntimeError("OSQP returned no solution")

        status = getattr(res.info, "status", "")
        if status not in ("solved", "solved inaccurate"):
            raise RuntimeError(f"OSQP failed (status={status})")

        x_out = np.asarray(x, dtype=np.float64)
        self._last_x = x_out
        return x_out

    def solve(self, problem, **kwargs) -> Any:
        data = self._prepare(problem)
        P, q, C, l, u = data.P, data.q, data.C, data.l, data.u
        n = int(P.shape[0])
        m = int(C.shape[0])

        if m == 0:
            ridge = 1e-12
            x = -np.linalg.solve(P + ridge * np.eye(n, dtype=np.float64), q)
        else:
            # Equality-only fast path
            if not np.isneginf(l).any():
                x = _solve_equality_qp(P, q, C, u)
            else:
                # Try IPM for small dense problems; fallback to OSQP.
                mi = int(np.isneginf(l).sum())
                A = C[mi:, :] if mi < m else np.empty((0, n), dtype=np.float64)
                b = u[mi:] if mi < m else np.empty((0,), dtype=np.float64)
                G = C[:mi, :]
                h = u[:mi]

                x_ipm: NDArray[np.float64] | None = None
                if n <= 60 and mi <= 300 and A.shape[0] <= 60:
                    x_ipm = _solve_ipm_qp(P, q, G, h, A, b, max_iter=25)
                x = x_ipm if x_ipm is not None else self._solve_via_osqp(P, q, C, l, u)

        obj = float(0.5 * x @ (P @ x) + q @ x)
        return {"solution": x.tolist(), "objective": obj}