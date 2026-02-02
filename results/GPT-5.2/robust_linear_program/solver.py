from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

try:
    import ecos  # type: ignore
except Exception:  # pragma: no cover
    ecos = None

# Cache CSC structure (indices/indptr) for dense G keyed by (total_rows, n)
_CSC_STRUCT_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
# Cache reusable (Gs, data_buffer) keyed by (m, n)
_G_CACHE: dict[tuple[int, int], tuple[sp.csc_matrix, np.ndarray]] = {}
# Cache reusable h buffer keyed by (m, n)
_H_CACHE: dict[tuple[int, int], np.ndarray] = {}
_DIMS_CACHE: dict[tuple[int, int], dict[str, Any]] = {}

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Robust LP with ellipsoidal uncertainty as SOCP:
            minimize c^T x
            s.t. ||P_i^T x||_2 <= b_i - q_i^T x  for i=1..m

        Solved via ECOS directly (cone form) for speed; with feasibility-checked
        fallback for rare numerical issues.
        """
        c = np.asarray(problem["c"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        n = int(c.shape[0])
        m = int(b.shape[0])

        if m == 0:
            if np.all(c == 0):
                x0 = np.zeros(n, dtype=np.float64)
                return {"objective_value": 0.0, "x": x0}
            return {"objective_value": float("inf"), "x": np.full(n, np.nan, dtype=np.float64)}

        # Fast path: typical inputs are well-formed nested lists / ndarrays
        P = np.asarray(problem["P"], dtype=np.float64)  # (m,n,n)
        q = np.asarray(problem["q"], dtype=np.float64)  # (m,n)

        if ecos is None:
            return {"objective_value": float("inf"), "x": np.full(n, np.nan, dtype=np.float64)}

        rows_per = n + 1
        total_rows = m * rows_per
        key_mn = (m, n)

        # Views (no copy)
        PT = P.swapaxes(1, 2)  # P_i^T, shape (m,n,n)

        # h: [b_i; 0..0] per constraint (reuse buffer)
        h = _H_CACHE.get(key_mn)
        if h is None or h.shape[0] != total_rows:
            h = np.zeros(total_rows, dtype=np.float64)
            _H_CACHE[key_mn] = h
        else:
            h.fill(0.0)
        h[0::rows_per] = b

        # Reuse a cached CSC matrix (structure fixed for given m,n), refill data only
        cached = _G_CACHE.get(key_mn)
        if cached is None:
            key = (total_rows, n)
            st = _CSC_STRUCT_CACHE.get(key)
            if st is None:
                indices = np.tile(np.arange(total_rows, dtype=np.int32), n)
                indptr = (np.arange(n + 1, dtype=np.int32) * total_rows).astype(np.int32, copy=False)
                _CSC_STRUCT_CACHE[key] = (indices, indptr)
            else:
                indices, indptr = st

            data = np.empty(total_rows * n, dtype=np.float64)
            Gs = sp.csc_matrix((data, indices, indptr), shape=(total_rows, n), copy=False)
            _G_CACHE[key_mn] = (Gs, data)
        else:
            Gs, data = cached

        # Fill G data in-place:
        # For each column j: top of each block is q[i,j], lower is -P[i,j,:]
        data3 = data.reshape((n, m, rows_per))  # (col j, constraint i, within-block row)
        data3[:, :, 0] = q.T
        data3[:, :, 1:] = -P.swapaxes(0, 1)  # (n,m,n) with [j,i,k]=P[i,j,k]

        dims = _DIMS_CACHE.get(key_mn)
        if dims is None:
            dims = {"l": 0, "q": [rows_per] * m, "s": []}
            _DIMS_CACHE[key_mn] = dims

        def _try_ecos(abstol: float, reltol: float, feastol: float, max_iters: int) -> np.ndarray | None:
            try:
                sol = ecos.solve(
                    c,
                    Gs,
                    h,
                    dims,
                    A=None,
                    b=None,
                    verbose=False,
                    abstol=abstol,
                    reltol=reltol,
                    feastol=feastol,
                    max_iters=max_iters,
                )
            except Exception:
                return None
            x = sol.get("x", None)
            if x is None:
                return None
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            if x.shape[0] != n or (not np.all(np.isfinite(x))):
                return None
            return x

        def _max_violation(x: np.ndarray) -> float:
            # viol_i = ||P_i^T x|| - (b_i - q_i^T x)
            Px = PT @ x  # (m,n)
            lhs = np.sqrt(np.sum(Px * Px, axis=1))
            rhs = b - (q @ x)
            return float(np.max(lhs - rhs))

        # First ECOS attempt: aggressive (most instances should pass; else fallback)
        x = _try_ecos(abstol=3e-6, reltol=3e-6, feastol=3e-6, max_iters=35)
        if x is not None and _max_violation(x) <= 1e-8:
            return {"objective_value": float(c @ x), "x": x}

        # Second ECOS attempt: tighter
        x = _try_ecos(abstol=1e-9, reltol=1e-9, feastol=1e-9, max_iters=200)
        if x is not None and _max_violation(x) <= 1e-8:
            return {"objective_value": float(c @ x), "x": x}

        # Last resort: CVXPY (slow, but should be very rare)
        try:
            import cvxpy as cp  # type: ignore

            xvar = cp.Variable(n)
            cons = []
            for i in range(m):
                cons.append(cp.SOC(b[i] - q[i].T @ xvar, PT[i] @ xvar))
            prob = cp.Problem(cp.Minimize(c.T @ xvar), cons)
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if prob.status not in ("optimal", "optimal_inaccurate") or xvar.value is None:
                return {"objective_value": float("inf"), "x": np.full(n, np.nan, dtype=np.float64)}
            xcvx = np.asarray(xvar.value, dtype=np.float64).reshape(-1)
            obj = float(c @ xcvx)
            if not np.isfinite(obj) or not np.all(np.isfinite(xcvx)):
                return {"objective_value": float("inf"), "x": np.full(n, np.nan, dtype=np.float64)}
            return {"objective_value": obj, "x": xcvx}
        except Exception:
            return {"objective_value": float("inf"), "x": np.full(n, np.nan, dtype=np.float64)}