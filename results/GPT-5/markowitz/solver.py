from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        # Parse input (accept both Greek and ASCII keys for robustness)
        mu_key = "μ" if "μ" in problem else ("mu" if "mu" in problem else "μ")
        sig_key = "Σ" if "Σ" in problem else ("Sigma" if "Sigma" in problem else "Σ")
        gamma_key = "γ" if "γ" in problem else ("gamma" if "gamma" in problem else "γ")

        mu = np.asarray(problem[mu_key], dtype=float)
        Sigma = np.asarray(problem[sig_key], dtype=float)
        gamma = float(problem[gamma_key])

        n = mu.size
        if n == 0:
            return {"w": []}
        if n == 1:
            return {"w": [1.0]}
        if not np.isfinite(mu).all() or not np.isfinite(Sigma).all() or not np.isfinite(gamma):
            w = np.zeros(n)
            w[int(np.nanargmax(mu))] = 1.0
            return {"w": w.tolist()}
        if gamma <= 0:
            w = np.zeros(n)
            w[int(np.argmax(mu))] = 1.0
            return {"w": w.tolist()}
        # Try fast diagonal closed-form
        w_diag = self._solve_diagonal(mu, Sigma, gamma)
        if w_diag is not None:
            return {"w": w_diag.tolist()}

        # General PSD case: fast KKT-based active-set on simplex
        w = self._active_set_markowitz(mu, Sigma, gamma)

        # Fallback to projected gradient if anything went wrong
        if (not np.isfinite(w).all()) or (w.min() < -1e-8) or (abs(w.sum() - 1.0) > 1e-8):
            w = self._projected_gradient(mu, Sigma, gamma, w0=None)

        # Final cleanup and normalization
        w = np.asarray(w, dtype=float)
        w[~np.isfinite(w)] = 0.0
        w[w < 0] = 0.0
        s = float(w.sum())
        if s <= 0:
            w[:] = 0.0
            w[int(np.argmax(mu))] = 1.0
        else:
            w /= s

        return {"w": w.tolist()}

    @staticmethod
    def _solve_diagonal(mu: np.ndarray, Sigma: np.ndarray, gamma: float) -> Optional[np.ndarray]:
        """Closed-form solver when Sigma is (nearly) diagonal.
        Returns None if Sigma is not sufficiently close to diagonal.
        """
        n = mu.size
        diag = np.diag(Sigma)
        # Compute maximum off-diagonal absolute value without forming a separate off-diagonal matrix
        A = np.abs(Sigma).copy()
        if n > 0:
            np.fill_diagonal(A, 0.0)
        max_off = float(A.max()) if A.size else 0.0
        max_all = float(np.max(np.abs(Sigma))) if Sigma.size else 0.0
        if max_off > 1e-14 * (1.0 + max_all):
            return None  # not diagonal enough

        sigma = diag.copy()
        pos_mask = sigma > 0.0
        zero_mask = ~pos_mask

        if pos_mask.sum() == 0:
            w = np.zeros(n)
            w[int(np.argmax(mu))] = 1.0
            return w

        # Water-filling solution
        a = np.zeros(n)
        a[pos_mask] = 1.0 / (2.0 * gamma * sigma[pos_mask])

        idx_pos = np.where(pos_mask)[0]
        mu_pos = mu[idx_pos]
        a_pos = a[idx_pos]
        order = np.argsort(-mu_pos)
        mu_sorted = mu_pos[order]
        a_sorted = a_pos[order]
        idx_sorted = idx_pos[order]

        s_prefix = np.cumsum(a_sorted * mu_sorted)
        d_prefix = np.cumsum(a_sorted)

        lam: Optional[float] = None
        m = idx_sorted.size
        mu_next = np.concatenate([mu_sorted[1:], np.array([-np.inf])])
        for k in range(m):
            d_k = d_prefix[k]
            if d_k <= 0:
                continue
            lam_k = (s_prefix[k] - 1.0) / d_k
            upper = mu_sorted[k]
            lower = mu_next[k]
            if (lam_k <= upper + 1e-15) and (lam_k >= lower - 1e-15):
                lam = float(lam_k)
                break
        if lam is None:
            d_k = d_prefix[-1]
            lam = float((s_prefix[-1] - 1.0) / (d_k if d_k > 0 else 1.0))

        if zero_mask.any():
            mu_zero_max = float(np.max(mu[zero_mask]))
            if lam < mu_zero_max:
                lam = mu_zero_max

        w = np.zeros(n)
        w[pos_mask] = a[pos_mask] * np.maximum(0.0, mu[pos_mask] - lam)

        leftover = 1.0 - float(w.sum())
        if leftover > 0.0 and zero_mask.any():
            idx_zero = np.where(zero_mask)[0]
            mu_zero = mu[idx_zero]
            close = np.where(np.abs(mu_zero - lam) <= 1e-12)[0]
            if close.size == 0:
                j = int(idx_zero[int(np.argmax(mu_zero))])
                w[j] += leftover
            else:
                share = leftover / close.size
                for c in close:
                    w[int(idx_zero[int(c)])] += share
        elif leftover < -1e-12:
            w = np.maximum(w, 0.0)
            s = float(w.sum())
            if s > 0:
                w /= s
            else:
                w[:] = 0.0
                w[int(np.argmax(mu))] = 1.0

        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s <= 0:
            w[:] = 0.0
            w[int(np.argmax(mu))] = 1.0
        else:
            w /= s
        return w

    @staticmethod
    def _active_set_markowitz(mu: np.ndarray, Sigma: np.ndarray, gamma: float) -> np.ndarray:
        """Fast KKT-based active-set solver for:
            maximize mu^T w - gamma * w^T Sigma w
            s.t. sum w = 1, w >= 0

        Maintains the inverse of the free-set Hessian to accelerate iterations.
        """
        n = mu.size
        if n == 0:
            return np.array([], dtype=float)

        # Define H = 2*gamma*Sigma for convenience
        H = 2.0 * gamma * Sigma

        # Initialize free set with best single asset
        diag_S = np.clip(np.diag(Sigma), 0.0, np.inf)
        start_idx = int(np.argmax(mu - gamma * diag_S))
        F: list[int] = [start_idx]
        inF = np.zeros(n, dtype=bool)
        inF[start_idx] = True

        w = np.zeros(n, dtype=float)

        # Numerical tolerances
        rc_tol = 1e-10
        neg_tol = 1e-12
        max_iter = max(50, 10 * n)

        # Regularization for numerical stability of inverse
        H_diag_max = float(np.max(np.abs(np.diag(H)))) if n > 0 else 1.0
        reg_base = 1e-12 * (1.0 + H_diag_max)

        # Maintain inverse of H_FF + reg*I (for current reg)
        M: Optional[np.ndarray] = None
        current_reg = reg_base

        def build_inverse(F_idx: np.ndarray, reg: float) -> tuple[np.ndarray, float]:
            """Build inverse of (HFF + reg*I) via Cholesky, escalating reg if needed."""
            k = F_idx.size
            if k == 0:
                return np.empty((0, 0), dtype=float), reg
            HFF = H[np.ix_(F_idx, F_idx)]
            reg_try = reg
            for _ in range(8):
                try:
                    HFF_reg = HFF + reg_try * np.eye(k)
                    L = np.linalg.cholesky(HFF_reg)
                    I = np.eye(k)
                    Y = np.linalg.solve(L, I)
                    M_local = np.linalg.solve(L.T, Y)
                    return M_local, reg_try
                except np.linalg.LinAlgError:
                    reg_try *= 10.0
            # As last resort, use pseudo-inverse
            HFF_reg = HFF + reg_try * np.eye(k)
            M_local = np.linalg.pinv(HFF_reg)
            return M_local, reg_try

        def add_index_update_inverse(M_old: np.ndarray, F_idx: np.ndarray, j: int, reg: float) -> tuple[np.ndarray, float]:
            """Update inverse when adding index j to the free set using block inverse.
            If update is unstable, rebuild from scratch.
            """
            k = F_idx.size
            if k == 0:
                hjj = float(H[j, j] + reg)
                if not np.isfinite(hjj) or hjj <= 0:
                    return build_inverse(np.array([j], dtype=int), reg)
                return np.array([[1.0 / hjj]], dtype=float), reg
            # u = H[F, j]
            u = H[F_idx, j]
            v = M_old @ u
            hjj_reg = float(H[j, j] + reg)
            s = hjj_reg - float(u.T @ v)
            if not np.isfinite(s) or s <= 1e-18:
                return build_inverse(np.append(F_idx, j), reg)
            inv_s = 1.0 / s
            # Compose new inverse
            TL = M_old + np.outer(v, v) * inv_s
            TR = -v * inv_s
            M_new = np.empty((k + 1, k + 1), dtype=float)
            M_new[:k, :k] = TL
            M_new[:k, k] = TR
            M_new[k, :k] = TR
            M_new[k, k] = inv_s
            return M_new, reg

        # Initialize inverse for first free set
        F_idx = np.array(F, dtype=int)
        M, current_reg = build_inverse(F_idx, reg_base)

        it = 0
        while it < max_iter:
            it += 1
            F_idx = np.array(F, dtype=int)
            k = F_idx.size
            onesk = np.ones(k, dtype=float)

            # Solve for w_F and lambda using M ≈ (HFF+reg I)^{-1}
            # x = M 1, y = M mu_F; then lambda = (1^T y - 1)/(1^T x), wF = y - lambda x
            if k == 0:
                break
            x = M @ onesk
            y = M @ mu[F_idx]
            denom = float(onesk @ x)
            if abs(denom) < 1e-18 or not np.isfinite(denom):
                # Rebuild with higher reg
                M, current_reg = build_inverse(F_idx, max(current_reg * 10.0, reg_base))
                x = M @ onesk
                y = M @ mu[F_idx]
                denom = float(onesk @ x)
                if abs(denom) < 1e-18 or not np.isfinite(denom):
                    break
            lam = (float(onesk @ y) - 1.0) / denom
            wF = y - lam * x

            # Remove negative components if any
            if wF.size and np.min(wF) <= -neg_tol:
                i_rm_local = int(np.argmin(wF))
                idx_rm = int(F_idx[i_rm_local])
                inF[idx_rm] = False
                # Remove and rebuild inverse for robustness
                F.pop(i_rm_local)
                F_idx = np.array(F, dtype=int)
                M, current_reg = build_inverse(F_idx, current_reg)
                continue

            if wF.size and np.any(wF < 0.0):
                negs = np.where(wF < 0.0)[0]
                for loc in negs[::-1]:
                    idx_rm = int(F_idx[int(loc)])
                    inF[idx_rm] = False
                    F.remove(idx_rm)
                F_idx = np.array(F, dtype=int)
                M, current_reg = build_inverse(F_idx, current_reg)
                continue

            # Compute reduced costs for inactive set: r = mu_B - lam - H[B,F] @ wF
            if k < n:
                B_idx = np.where(~inF)[0]
                if B_idx.size > 0:
                    v = H[np.ix_(B_idx, F_idx)].dot(wF) if k > 0 else np.zeros(B_idx.size)
                    r = mu[B_idx] - lam - v
                    i_add_local = int(np.argmax(r))
                    max_r = float(r[i_add_local])
                    if max_r > rc_tol:
                        idx_add = int(B_idx[i_add_local])
                        F.append(idx_add)
                        inF[idx_add] = True
                        # Fast inverse update for addition
                        old_F_idx = np.array(F[:-1], dtype=int)
                        M, current_reg = add_index_update_inverse(M, old_F_idx, idx_add, current_reg)
                        continue

            # Optimal, assemble result
            w[:] = 0.0
            if k > 0:
                w[F_idx] = wF
            break

        # Cleanup and normalization
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s <= 0:
            w[:] = 0.0
            w[int(np.argmax(mu))] = 1.0
        else:
            w /= s
        return w

    @staticmethod
    def _projected_gradient(
        mu: np.ndarray,
        Sigma: np.ndarray,
        gamma: float,
        w0: Optional[np.ndarray] = None,
        max_iter: int = 4000,
    ) -> np.ndarray:
        """Projected gradient ascent on the simplex for g(w) = mu^T w - gamma * w^T Sigma w."""
        n = mu.size
        if w0 is None or w0.shape != (n,) or not np.isfinite(w0).all():
            w = np.ones(n, dtype=float) / n
        else:
            w = w0.copy()
            w[~np.isfinite(w)] = 0.0
            w[w < 0] = 0.0
            s = float(w.sum())
            w = (w / s) if s > 0 else np.ones(n) / n

        # Lipschitz bound for gradient: L <= ||2*gamma*Sigma||_2 approximated by row-sum norm
        H = 2.0 * gamma * Sigma
        H_inf = float(np.max(np.sum(np.abs(H), axis=1)))
        L = max(1.0, H_inf)
        t = 1.0 / L

        def proj_simplex(v: np.ndarray) -> np.ndarray:
            u = np.sort(v)[::-1]
            cssv = np.cumsum(u)
            rho = np.nonzero(u - (cssv - 1) / (np.arange(v.size) + 1) > 0)[0]
            if rho.size == 0:
                theta = (cssv[-1] - 1) / v.size
            else:
                rho = int(rho[-1])
                theta = (cssv[rho] - 1) / float(rho + 1)
            w_proj = v - theta
            np.maximum(w_proj, 0.0, out=w_proj)
            return w_proj

        g_prev = -np.inf
        for _ in range(max_iter):
            grad = mu - H.dot(w)
            w_new = proj_simplex(w + t * grad)
            g = float(mu @ w_new - gamma * (w_new @ Sigma @ w_new))
            if g + 1e-12 < g_prev:
                t *= 0.5
                if t < 1e-18:
                    break
                continue
            if np.linalg.norm(w_new - w, ord=1) < 1e-12:
                w = w_new
                break
            w = w_new
            g_prev = g

        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s <= 0:
            w[:] = 0.0
            w[int(np.argmax(mu))] = 1.0
        else:
            w /= s
        return w