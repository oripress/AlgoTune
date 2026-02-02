from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    """
    Long-only Markowitz portfolio optimization:

        maximize    μ^T w - γ * w^T Σ w
        subject to  1^T w = 1
                    w >= 0

    Active-set over nonnegativity constraints. On a support A (w_A > 0),
    KKT system is:

        (2γ Σ_AA) w_A + λ 1 = μ_A
        1^T w_A = 1

    We update the support until primal/dual feasibility holds.
    """

    @staticmethod
    def _solve_support(
        Sigma: np.ndarray, mu: np.ndarray, two_g: float, A: np.ndarray
    ) -> tuple[np.ndarray, float] | None:
        """Solve KKT on support A; returns (w_A, lambda)."""
        k = int(A.size)
        if k <= 0:
            return None
        if k == 1:
            i = int(A[0])
            wA = np.array([1.0], dtype=np.float64)
            lam = float(mu[i] - two_g * Sigma[i, i])  # stationarity on active var
            return wA, lam

        S_AA = Sigma[np.ix_(A, A)]
        Q = two_g * S_AA
        ones = np.ones(k, dtype=np.float64)
        mu_A = mu[A]

        # Try PD solve via Cholesky (fast); otherwise fall back to full KKT solve.
        try:
            L = np.linalg.cholesky(Q)
            rhs = np.empty((k, 2), dtype=np.float64)
            rhs[:, 0] = mu_A
            rhs[:, 1] = ones
            y = np.linalg.solve(L, rhs)
            x = np.linalg.solve(L.T, y)
            invQ_mu = x[:, 0]
            invQ_1 = x[:, 1]
            denom = float(ones @ invQ_1)
            if denom == 0.0 or not np.isfinite(denom):
                raise np.linalg.LinAlgError
            lam = float((ones @ invQ_mu - 1.0) / denom)
            wA = invQ_mu - lam * invQ_1
            return wA, lam
        except Exception:
            K = np.empty((k + 1, k + 1), dtype=np.float64)
            K[:k, :k] = Q
            K[:k, k] = 1.0
            K[k, :k] = 1.0
            K[k, k] = 0.0

            rhs = np.empty(k + 1, dtype=np.float64)
            rhs[:k] = mu_A
            rhs[k] = 1.0

            try:
                sol = np.linalg.solve(K, rhs)
            except Exception:
                sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
            return sol[:k], float(sol[k])

    @staticmethod
    def _solve_diagonal(mu: np.ndarray, diag: np.ndarray, gamma: float) -> np.ndarray:
        """Exact solver when Σ is (approximately) diagonal."""
        n = int(mu.size)
        two_g = 2.0 * gamma
        a = two_g * diag  # a_i >= 0

        pos = a > 0.0
        zero = ~pos

        w = np.zeros(n, dtype=np.float64)

        def waterfill(mu_p: np.ndarray, a_p: np.ndarray) -> np.ndarray:
            m = int(mu_p.size)
            order = np.argsort(mu_p)[::-1]
            mu_s = mu_p[order]
            inva_s = 1.0 / a_p[order]
            cum_inva = np.cumsum(inva_s)
            cum_mu_inva = np.cumsum(mu_s * inva_s)

            lam = None
            for t in range(1, m + 1):
                denom = float(cum_inva[t - 1])
                num = float(cum_mu_inva[t - 1] - 1.0)
                lt = num / denom
                if lt < float(mu_s[t - 1]) + 1e-14 and (t == m or lt >= float(mu_s[t]) - 1e-14):
                    lam = lt
                    t_star = t
                    break
            if lam is None:
                t_star = m
                lam = float((cum_mu_inva[-1] - 1.0) / cum_inva[-1])

            w_s = np.zeros(m, dtype=np.float64)
            w_s[:t_star] = (mu_s[:t_star] - lam) * inva_s[:t_star]
            w_p = np.zeros(m, dtype=np.float64)
            w_p[order] = w_s
            return w_p

        if np.any(zero):
            # Candidate riskless asset (a_i == 0). If it ends up with positive weight, λ = μ0.
            idx0 = int(np.flatnonzero(zero)[np.argmax(mu[zero])])
            mu0 = float(mu[idx0])

            if np.any(pos):
                w_pos = np.maximum(0.0, (mu[pos] - mu0) / a[pos])
                s = float(w_pos.sum())
                if s <= 1.0 + 1e-12:
                    w[pos] = w_pos
                    w[idx0] = max(0.0, 1.0 - s)
                    # normalize to be safe
                    ss = float(w.sum())
                    if ss > 0.0:
                        w /= ss
                    return w

            # Otherwise riskless excluded; solve on pos with sum=1.
            if np.any(pos):
                w[pos] = waterfill(mu[pos], a[pos])
                ss = float(w.sum())
                if ss > 0.0:
                    w /= ss
                return w

            # All a_i == 0 => pure LP on simplex => best μ.
            w[idx0] = 1.0
            return w

        # Standard diagonal case (all a_i > 0)
        w[:] = waterfill(mu, a)
        ss = float(w.sum())
        if ss > 0.0:
            w /= ss
        return w

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, list[float]] | None:
        mu = np.asarray(problem["μ"], dtype=np.float64)
        Sigma = np.asarray(problem["Σ"], dtype=np.float64)
        gamma = float(problem["γ"])

        n = int(mu.size)
        if n <= 0:
            return None
        if n == 1:
            return {"w": [1.0]}
        if gamma <= 0.0 or not np.isfinite(gamma):
            return None
        if not np.isfinite(mu).all() or not np.isfinite(Sigma).all():
            return None

        # Very small risk term => best-return vertex.
        sig_norm = float(np.max(np.abs(Sigma)))
        if not np.isfinite(sig_norm):
            return None
        if gamma * sig_norm < 1e-14:
            j = int(np.argmax(mu))
            w0 = np.zeros(n, dtype=np.float64)
            w0[j] = 1.0
            return {"w": w0.tolist()}

        # Fast diagonal path (only if matrix is very sparse -> likely diagonal).
        nnz = int(np.count_nonzero(Sigma))
        if nnz <= 6 * n:
            off = Sigma.copy()
            np.fill_diagonal(off, 0.0)
            if float(np.max(np.abs(off))) <= 1e-15 * (float(np.max(np.abs(np.diagonal(Sigma)))) + 1.0):
                wdiag = self._solve_diagonal(mu, np.diagonal(Sigma).copy(), gamma)
                return {"w": wdiag.tolist()}

        two_g = 2.0 * gamma

        in_support = np.ones(n, dtype=bool)

        tol_w_remove = 1e-12
        tol_dual = 1e-12
        max_iter = 4 * n + 50

        w = np.full(n, 1.0 / n, dtype=np.float64)
        lam = 0.0

        for _ in range(max_iter):
            A = np.flatnonzero(in_support)
            sol = self._solve_support(Sigma, mu, two_g, A)
            if sol is None:
                break
            wA, lam = sol

            w[:] = 0.0
            w[A] = wA

            # If any weights are negative, drop all sufficiently negative and retry.
            neg = wA < -tol_w_remove
            if np.any(neg):
                in_support[A[neg]] = False
                continue

            # Dual feasibility for inactive variables:
            # s_i = 2γ(Σw)_i - μ_i + λ >= 0 for w_i = 0
            k = int(A.size)
            if k * 4 < n:
                # Use only active columns: Σw = Σ[:,A] wA  (O(nk))
                Sw = Sigma[:, A] @ wA
            else:
                Sw = Sigma @ w
            s = two_g * Sw - mu
            s += lam

            I = ~in_support
            if np.any(I):
                idxI = np.flatnonzero(I)
                j = int(idxI[int(np.argmin(s[idxI]))])
                if float(s[j]) < -tol_dual:
                    in_support[j] = True
                    continue

            break
        else:
            # Rare fallback to CVXPY for robustness.
            try:
                import cvxpy as cp

                wv = cp.Variable(n)
                obj = cp.Maximize(mu @ wv - gamma * cp.quad_form(wv, cp.psd_wrap(Sigma)))
                cons = [cp.sum(wv) == 1, wv >= 0]
                cp.Problem(obj, cons).solve(solver=cp.ECOS, warm_start=True, verbose=False)
                if wv.value is None:
                    return None
                w = np.asarray(wv.value, dtype=np.float64)
            except Exception:
                return None

        if not np.isfinite(w).all():
            return None
        w[w < 0.0] = 0.0
        ssum = float(w.sum())
        if not np.isfinite(ssum) or ssum <= 0.0:
            return None
        w /= ssum
        return {"w": w.tolist()}