from typing import Any, Dict, List
import numpy as np

# Try to import SciPy's linprog (HiGHS) for a fast LP solver
try:
    from scipy.optimize import linprog  # type: ignore
    _HAS_LINPROG = True
except Exception:
    linprog = None  # type: ignore
    _HAS_LINPROG = False

# Try to import scipy.sparse to build sparse constraint matrices when helpful
try:
    import scipy.sparse as sp  # type: ignore
    _HAS_SPARSE = True
except Exception:
    sp = None  # type: ignore
    _HAS_SPARSE = False

class Solver:
    def _ensure_array2d(self, a_in: Any) -> np.ndarray:
        a = np.asarray(a_in, dtype=float)
        if a.ndim == 0 or a.size == 0:
            return np.zeros((0, 0), dtype=float)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        elif a.ndim > 2:
            a = a.reshape(a.shape[0], -1)
        return a

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        """
        Fast solver for the Chebyshev center problem:
            maximize r
            s.t. a_i^T x + r * ||a_i|| <= b_i, i=1..m
                 r >= 0

        Strategy:
        - Normalize each constraint by its Euclidean norm so it becomes:
            (a_i / ||a_i||)^T x + r <= b_i / ||a_i||
          This makes the r-column simply ones and improves numeric stability.
        - Solve the resulting primal LP (variables [x, r]) using SciPy's HiGHS (linprog).
        - If the primal solve fails, attempt the dual (fewer variables when m << n),
          reconstruct a primal candidate from active dual multipliers, and validate.
        - Fallback to a least-squares heuristic if all else fails.
        """
        # Basic validation and conversions
        if not isinstance(problem, dict):
            return {"solution": []}
        a_in = problem.get("a")
        b_in = problem.get("b")
        if a_in is None or b_in is None:
            return {"solution": []}

        a = self._ensure_array2d(a_in)
        if a.size == 0:
            # No constraints; ambiguous. Return empty or zero-d vector if n known.
            return {"solution": []}
        m, n = a.shape

        try:
            b = np.asarray(b_in, dtype=float).reshape(-1)
        except Exception:
            b = np.zeros(m, dtype=float)

        # Adjust b length to match m
        if b.size != m:
            if b.size == 0:
                b = np.zeros(m, dtype=float)
            elif b.size == 1:
                b = np.full(m, float(b.item()), dtype=float)
            elif b.size < m:
                tmp = np.zeros(m, dtype=float)
                tmp[: b.size] = b
                b = tmp
            else:
                b = b[:m]

        # Quick return for trivial case
        if m == 0:
            return {"solution": [0.0] * n}

        # Compute row norms and handle zero-norm constraints
        norms = np.linalg.norm(a, axis=1)
        tol_zero = 1e-12
        nonzero_mask = norms > tol_zero

        # If any zero-norm row has b < 0, the system is infeasible -> return zeros
        if np.any(~nonzero_mask):
            if np.any(b[~nonzero_mask] < -1e-12):
                # Infeasible; no valid center. Return a stable fallback.
                return {"solution": [0.0] * n}
            # Drop irrelevant zero rows (0^T x + 0 * r <= b with b >= 0)
            if np.all(nonzero_mask):
                pass
            else:
                a = a[nonzero_mask]
                b = b[nonzero_mask]
                norms = norms[nonzero_mask]
                m = a.shape[0]
                if m == 0:
                    # Only trivial constraints existed; return zero center.
                    return {"solution": [0.0] * n}

        # Normalize rows so that coefficient on r becomes 1
        a_hat = a / norms.reshape(-1, 1)
        b_hat = b / norms

        # Build primal A_ub = [a_hat, 1] and b_ub = b_hat
        ones_col = np.ones((m, 1), dtype=float)
        use_sparse = _HAS_SPARSE and (m * (n + 1) > 200_000)
        if use_sparse:
            try:
                A_left = sp.csr_matrix(a_hat)
                A_right = sp.csr_matrix(ones_col)
                A_ub = sp.hstack([A_left, A_right], format="csr")
            except Exception:
                A_ub = np.hstack([a_hat, ones_col])
        else:
            A_ub = np.hstack([a_hat, ones_col])
        b_ub = b_hat

        # Objective: minimize -r  (equivalent to maximize r)
        c_primal = np.zeros(n + 1, dtype=float)
        c_primal[-1] = -1.0
        bounds_primal = [(None, None)] * n + [(0.0, None)]

        # Try primal solve first (usually fastest and robust)
        if _HAS_LINPROG:
            try:
                res = linprog(c_primal, A_ub=A_ub, b_ub=b_ub, bounds=bounds_primal, method="highs")
                if getattr(res, "success", False) and getattr(res, "x", None) is not None:
                    z = np.asarray(res.x, dtype=float)
                    x = z[:n]
                    x = np.nan_to_num(x, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
                    return {"solution": x.tolist()}
            except Exception:
                # Fall through to fallback strategies
                pass

        # Fallback: try solving the dual (may be faster when m << n) and reconstruct primal
        # Dual (for maximize r formulation) is:
        #   minimize b_hat^T y  subject to a_hat^T y = 0, 1^T y = 1, y >= 0
        if _HAS_LINPROG:
            try:
                c_dual = b_hat.copy()
                # Build A_eq: first n rows are a_hat.T, last row is ones
                if use_sparse:
                    try:
                        Aeq_top = sp.csr_matrix(a_hat.T)
                        Aeq_last = sp.csr_matrix(np.ones((1, m), dtype=float))
                        A_eq = sp.vstack([Aeq_top, Aeq_last], format="csr")
                    except Exception:
                        A_eq = np.vstack([a_hat.T, np.ones((1, m), dtype=float)])
                else:
                    A_eq = np.vstack([a_hat.T, np.ones((1, m), dtype=float)])
                b_eq = np.concatenate([np.zeros(n, dtype=float), np.array([1.0])])
                bounds_dual = [(0.0, None)] * m

                res2 = linprog(c_dual, A_eq=A_eq, b_eq=b_eq, bounds=bounds_dual, method="highs")
                if getattr(res2, "success", False) and getattr(res2, "x", None) is not None:
                    y = np.asarray(res2.x, dtype=float)

                    # Reconstruct primal (x, r) using active constraints
                    tol_active = 1e-9
                    active_idx = np.nonzero(y > tol_active)[0]
                    if active_idx.size == 0:
                        top_k = min(m, n + 1)
                        active_idx = np.argsort(-y)[:top_k]
                    if active_idx.size > (n + 1):
                        active_idx = active_idx[: (n + 1)]

                    A_top = np.hstack([a_hat[active_idx, :], np.ones((active_idx.size, 1), dtype=float)])
                    b_top = b_hat[active_idx]

                    # Solve least squares for [x; r]
                    z, *_ = np.linalg.lstsq(A_top, b_top, rcond=None)
                    z = np.asarray(z).flatten()
                    if z.size < n + 1:
                        z = np.concatenate([z, np.zeros(n + 1 - z.size)])
                    x = z[:n]
                    r_val = float(z[-1])

                    # Validate feasibility (allow small tolerances)
                    if r_val >= -1e-9 and np.all(a_hat.dot(x) + r_val <= b_hat + 1e-7):
                        x = np.nan_to_num(x, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
                        return {"solution": x.tolist()}
            except Exception:
                pass

        # Last-resort heuristic: least squares to approximate a solution
        try:
            x_ls, *_ = np.linalg.lstsq(a, b, rcond=None)
            x = np.asarray(x_ls).flatten()
            if x.size < n:
                x = np.concatenate([x, np.zeros(n - x.size, dtype=float)])
            x = np.nan_to_num(x, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
            return {"solution": x.tolist()}
        except Exception:
            return {"solution": [0.0] * n}