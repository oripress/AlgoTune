from __future__ import annotations

from typing import Any, Dict

import numpy as np

class Solver:
    """
    Fast projection onto {x : CVaR_beta(Ax) <= kappa}, with CVaR implemented as
    mean of the k = int((1-beta)*m) largest scenario losses (same as reference).

    Strategy:
      1) quick feasibility check for x0
      2) solve equivalent QP via HiGHS (highspy) with epigraph formulation of sum_largest
         sum_largest(Ax,k) <= alpha  <=>  exists t,u>=0:
             u_i >= a_i^T x - t
             k t + sum u_i <= alpha
      3) fallback to CVXPY only if HiGHS unavailable/fails
    """

    def __init__(self) -> None:
        self._checked_highs = False
        self._has_highs = False

    @staticmethod
    def _topk_mean(losses: np.ndarray, k: int) -> float:
        if k <= 0:
            return float("-inf")
        m = losses.size
        if k >= m:
            return float(np.mean(losses))
        part = np.partition(losses, m - k)
        return float(np.mean(part[m - k :]))

    def _ensure_highs(self) -> None:
        if self._checked_highs:
            return
        self._checked_highs = True
        try:
            import highspy  # noqa: F401  # type: ignore

            self._has_highs = True
        except Exception:
            self._has_highs = False

    def _solve_with_highs(self, x0: np.ndarray, A: np.ndarray, k: int, alpha: float) -> np.ndarray | None:
        self._ensure_highs()
        if not self._has_highs:
            return None

        import highspy  # type: ignore

        m, n = A.shape
        N = n + 1 + m  # x, t, u
        num_row = m + 1
        inf = float(getattr(highspy, "kHighsInf", 1e30))

        # Objective: x^T x - 2 x0^T x + const
        col_cost = np.zeros(N, dtype=np.float64)
        col_cost[:n] = -2.0 * x0

        col_lower = np.empty(N, dtype=np.float64)
        col_upper = np.empty(N, dtype=np.float64)
        col_lower[: n + 1] = -inf
        col_upper[: n + 1] = inf
        col_lower[n + 1 :] = 0.0
        col_upper[n + 1 :] = inf

        # Rows:
        # i=0..m-1: -A_i x + t + u_i >= 0
        # row m: k*t + sum(u) <= alpha
        row_lower = np.empty(num_row, dtype=np.float64)
        row_upper = np.empty(num_row, dtype=np.float64)
        row_lower[:m] = 0.0
        row_upper[:m] = inf
        row_lower[m] = -inf
        row_upper[m] = float(alpha)

        # Constraint matrix in CSC colwise.
        nnz = n * m + 3 * m + 1
        starts = np.empty(N + 1, dtype=np.int64)
        indices = np.empty(nnz, dtype=np.int32)
        values = np.empty(nnz, dtype=np.float64)

        p = 0
        starts[0] = 0
        row_ix = np.arange(m, dtype=np.int32)

        # x columns
        for j in range(n):
            indices[p : p + m] = row_ix
            values[p : p + m] = -A[:, j]
            p += m
            starts[j + 1] = p

        # t column
        indices[p : p + m] = row_ix
        values[p : p + m] = 1.0
        p += m
        indices[p] = m
        values[p] = float(k)
        p += 1
        starts[n + 1] = p

        # u columns
        for i in range(m):
            indices[p] = i
            values[p] = 1.0
            p += 1
            indices[p] = m
            values[p] = 1.0
            p += 1
            starts[n + 2 + i] = p

        if p != nnz:
            return None

        lp = highspy.HighsLp()
        lp.num_col_ = int(N)
        lp.num_row_ = int(num_row)
        lp.col_cost_ = col_cost.tolist()
        lp.col_lower_ = col_lower.tolist()
        lp.col_upper_ = col_upper.tolist()
        lp.row_lower_ = row_lower.tolist()
        lp.row_upper_ = row_upper.tolist()
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = starts.tolist()
        lp.a_matrix_.index_ = indices.tolist()
        lp.a_matrix_.value_ = values.tolist()

        # Hessian: 2*I on x block only (triangular storage)
        h_start = [0]
        h_index: list[int] = []
        h_value: list[float] = []
        for j in range(N):
            if j < n:
                h_index.append(j)
                h_value.append(2.0)
            h_start.append(len(h_index))

        hess = highspy.HighsHessian()
        hess.dim_ = int(N)
        hess.format_ = highspy.HessianFormat.kTriangular
        hess.start_ = h_start
        hess.index_ = h_index
        hess.value_ = h_value

        highs = highspy.Highs()
        try:
            highs.setOptionValue("output_flag", False)
        except Exception:
            pass

        highs.passModel(lp)
        highs.passHessian(hess)
        highs.run()

        try:
            sol = highs.getSolution()
            col_vals = np.asarray(sol.col_value, dtype=np.float64)
        except Exception:
            return None

        if col_vals.size < n:
            return None
        x = col_vals[:n].copy()
        if not np.all(np.isfinite(x)):
            return None
        return x

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        x0 = np.asarray(problem["x0"], dtype=np.float64)
        A = np.asarray(problem["loss_scenarios"], dtype=np.float64)
        beta = float(problem.get("beta", 0.95))
        kappa = float(problem.get("kappa", 0.0))

        m, n = A.shape
        k = int((1.0 - beta) * m)
        if k <= 0:
            return {"x_proj": x0.tolist()}

        # Fast skip if already feasible.
        cvar0 = self._topk_mean(A @ x0, k)
        if cvar0 <= kappa + 1e-12:
            return {"x_proj": x0.tolist()}

        alpha = kappa * k
        x = self._solve_with_highs(x0, A, k, alpha)

        if x is None:
            # Fallback to CVXPY epigraph (still avoids sum_largest).
            try:
                import cvxpy as cp  # type: ignore

                xvar = cp.Variable(n)
                t = cp.Variable()
                u = cp.Variable(m)
                constraints = [u >= 0, u >= A @ xvar - t, k * t + cp.sum(u) <= alpha]
                obj = cp.Minimize(cp.sum_squares(xvar - x0))
                prob = cp.Problem(obj, constraints)
                prob.solve(warm_start=True, solver=cp.ECOS)
                if xvar.value is None:
                    return {"x_proj": []}
                x = np.asarray(xvar.value, dtype=np.float64)
            except Exception:
                return {"x_proj": []}

        return {"x_proj": x.tolist()}