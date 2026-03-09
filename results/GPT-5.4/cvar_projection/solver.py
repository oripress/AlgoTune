from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, minimize

class Solver:
    def __init__(self):
        self._feas_tol = 1e-6

    @staticmethod
    def _topk_sum(values: np.ndarray, k: int) -> float:
        if k <= 0:
            return 0.0
        m = values.shape[0]
        if k >= m:
            return float(np.sum(values))
        idx = m - k
        return float(np.sum(np.partition(values, idx)[idx:]))

    @staticmethod
    def _topk_indices(values: np.ndarray, k: int) -> np.ndarray:
        m = values.shape[0]
        if k <= 0:
            return np.empty(0, dtype=int)
        if k >= m:
            return np.arange(m, dtype=int)
        idx = m - k
        return np.argpartition(values, idx)[idx:]

    @staticmethod
    def _topk_threshold(values: np.ndarray, k: int) -> float:
        if values.size == 0:
            return 0.0
        if k <= 0:
            return float(np.max(values))
        if k >= values.shape[0]:
            return float(np.min(values))
        idx = values.shape[0] - k
        return float(np.partition(values, idx)[idx])

    def _is_feasible(self, x: np.ndarray, A: np.ndarray, alpha: float, k: int) -> bool:
        return self._topk_sum(A @ x, k) <= alpha + self._feas_tol

    def _topk_cut(self, losses: np.ndarray, A: np.ndarray, k: int):
        idx = self._topk_indices(losses, k)
        top_sum = float(np.sum(losses[idx]))
        m = losses.shape[0]
        if k >= m:
            return top_sum, np.sum(A, axis=0), ("all",)

        tau = float(np.min(losses[idx]))
        tol = 1e-12
        gt_mask = losses > tau + tol
        num_gt = int(np.sum(gt_mask))
        rem = k - num_gt

        if rem > 0:
            eq_mask = np.abs(losses - tau) <= tol
            eq_count = int(np.sum(eq_mask))
            if 0 < rem < eq_count:
                cut = np.sum(A[gt_mask], axis=0) + (rem / eq_count) * np.sum(A[eq_mask], axis=0)
                return top_sum, cut, None

        return top_sum, np.sum(A[idx], axis=0), tuple(np.sort(idx).tolist())

    def _project_intersection(
        self,
        B: np.ndarray,
        x0: np.ndarray,
        alpha: float,
        lam_init: Optional[np.ndarray] = None,
    ):
        r = B.shape[0]
        if r == 0:
            return x0.copy(), np.zeros(0, dtype=float)

        G = B @ B.T
        c = B @ x0 - alpha

        if lam_init is None or lam_init.shape[0] != r:
            lam = np.zeros(r, dtype=float)
        else:
            lam = np.maximum(np.asarray(lam_init, dtype=float), 0.0)

        passive = lam > 1e-12

        for _ in range(5 * r + 20):
            grad = c - G @ lam
            inactive = ~passive
            if not np.any(inactive):
                break
            idx = np.flatnonzero(inactive)
            best = idx[int(np.argmax(grad[idx]))]
            if grad[best] <= 1e-10:
                break

            passive[best] = True
            for _ in range(5 * r + 20):
                p = np.flatnonzero(passive)
                z = np.zeros(r, dtype=float)
                if p.size:
                    sol, *_ = np.linalg.lstsq(G[np.ix_(p, p)], c[p], rcond=None)
                    z[p] = sol

                if p.size == 0 or np.all(z[p] > 1e-12):
                    lam = z
                    break

                bad = p[z[p] <= 1e-12]
                denom = lam[bad] - z[bad]
                step = np.min(lam[bad] / denom)
                lam = lam + step * (z - lam)
                lam[lam < 1e-12] = 0.0
                passive = lam > 1e-12

        lam = np.maximum(lam, 0.0)
        return x0 - B.T @ lam, lam

    def _solve_fast(self, x0: np.ndarray, A: np.ndarray, alpha: float, k: int):
        x = x0.copy()
        cuts = []
        keys = set()
        lam = np.zeros(0, dtype=float)

        for _ in range(64):
            losses = A @ x
            top_sum, cut, key = self._topk_cut(losses, A, k)
            if top_sum <= alpha + 1e-7:
                return x

            if key is not None and key in keys:
                return None
            if any(np.allclose(cut, old, rtol=1e-10, atol=1e-12) for old in cuts):
                return None

            if key is not None:
                keys.add(key)
            cuts.append(cut)
            B = np.asarray(cuts, dtype=float)

            lam0 = np.zeros(B.shape[0], dtype=float)
            if lam.size:
                lam0[: lam.size] = lam
            x, lam = self._project_intersection(B, x0, alpha, lam0)

        return None

    def _solve_qp(self, x0: np.ndarray, A: np.ndarray, alpha: float, k: int):
        m, n = A.shape
        nvar = n + 1 + m
        t_idx = n
        s0_idx = n + 1

        losses0 = A @ x0
        t0 = self._topk_threshold(losses0, k)
        s0 = np.maximum(losses0 - t0, 0.0)

        z0 = np.empty(nvar, dtype=float)
        z0[:n] = x0
        z0[t_idx] = t0
        z0[s0_idx:] = s0

        A_block = sp.csr_matrix(A)
        t_col = sp.csr_matrix(
            (-np.ones(m, dtype=float), (np.arange(m, dtype=int), np.zeros(m, dtype=int))),
            shape=(m, 1),
        )
        s_block = -sp.eye(m, format="csr")
        first = sp.hstack((A_block, t_col, s_block), format="csr")

        last_cols = np.concatenate(([t_idx], np.arange(s0_idx, nvar, dtype=int)))
        last_data = np.concatenate(([float(k)], np.ones(m, dtype=float)))
        last = sp.csr_matrix(
            (last_data, (np.zeros(m + 1, dtype=int), last_cols)),
            shape=(1, nvar),
        )

        mat = sp.vstack((first, last), format="csr")
        ub = np.zeros(m + 1, dtype=float)
        ub[m] = alpha
        constraint = LinearConstraint(mat, -np.inf, ub)

        lb = np.full(nvar, -np.inf, dtype=float)
        ub_b = np.full(nvar, np.inf, dtype=float)
        lb[s0_idx:] = 0.0
        bounds = Bounds(lb, ub_b)

        hdiag = np.concatenate((np.full(n, 2.0), np.zeros(1 + m, dtype=float)))

        def obj(z):
            d = z[:n] - x0
            return float(np.dot(d, d))

        def jac(z):
            g = np.zeros(nvar, dtype=float)
            g[:n] = 2.0 * (z[:n] - x0)
            return g

        x_candidate = None

        if m <= 200:
            try:
                dense_constraint = LinearConstraint(mat.toarray(), -np.inf, ub)
                res = minimize(
                    obj,
                    z0,
                    method="SLSQP",
                    jac=jac,
                    bounds=bounds,
                    constraints=(dense_constraint,),
                    options={"maxiter": 200, "ftol": 1e-9, "disp": False},
                )
                if res.x is not None:
                    x_candidate = np.asarray(res.x[:n], dtype=float)
            except Exception:
                x_candidate = None

        if x_candidate is None or not np.all(np.isfinite(x_candidate)) or not self._is_feasible(
            x_candidate, A, alpha, k
        ):
            try:
                res = minimize(
                    obj,
                    z0,
                    method="trust-constr",
                    jac=jac,
                    hess=lambda _z: sp.diags(hdiag, format="csr"),
                    bounds=bounds,
                    constraints=(constraint,),
                    options={"maxiter": 300, "gtol": 1e-8, "xtol": 1e-10, "verbose": 0},
                )
                if res.x is not None:
                    x_candidate = np.asarray(res.x[:n], dtype=float)
            except Exception:
                x_candidate = None

        if x_candidate is not None and np.all(np.isfinite(x_candidate)) and self._is_feasible(
            x_candidate, A, alpha, k
        ):
            return {"x_proj": x_candidate.tolist()}
        return None

    def _fallback_cvxpy(self, x0: np.ndarray, A: np.ndarray, alpha: float, k: int):
        try:
            import cvxpy as cp

            x = cp.Variable(x0.shape[0])
            prob = cp.Problem(cp.Minimize(cp.sum_squares(x - x0)), [cp.sum_largest(A @ x, k) <= alpha])
            prob.solve(warm_start=True)
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
                return {"x_proj": []}
            return {"x_proj": np.asarray(x.value, dtype=float).tolist()}
        except Exception:
            return {"x_proj": []}

    def solve(self, problem, **kwargs):
        x0 = np.asarray(problem["x0"], dtype=float)
        A = np.asarray(problem["loss_scenarios"], dtype=float)
        beta = float(problem["beta"])
        kappa = float(problem["kappa"])

        if x0.ndim != 1 or A.ndim != 2 or A.shape[1] != x0.shape[0]:
            return {"x_proj": []}

        m, _n = A.shape
        k = int((1.0 - beta) * m)
        alpha = kappa * k

        if m == 0 or k <= 0:
            return {"x_proj": x0.tolist()}

        if self._is_feasible(x0, A, alpha, k):
            return {"x_proj": x0.tolist()}

        if k == m:
            normal = np.sum(A, axis=0)
            denom = float(np.dot(normal, normal))
            if denom > 0.0:
                step = (float(np.dot(normal, x0)) - alpha) / denom
                if step >= 0.0:
                    x_half = x0 - step * normal
                    if self._is_feasible(x_half, A, alpha, k):
                        return {"x_proj": x_half.tolist()}

        x_fast = self._solve_fast(x0, A, alpha, k)
        if x_fast is not None and np.all(np.isfinite(x_fast)) and self._is_feasible(
            x_fast, A, alpha, k
        ):
            return {"x_proj": x_fast.tolist()}

        qp_result = self._solve_qp(x0, A, alpha, k)
        if qp_result is not None:
            return qp_result

        return self._fallback_cvxpy(x0, A, alpha, k)