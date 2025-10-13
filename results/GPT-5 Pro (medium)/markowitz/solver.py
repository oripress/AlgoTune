from __future__ import annotations

from typing import Any

import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[float]] | None:
        # Parse inputs with some robustness on key names
        mu = problem.get("μ", None)
        if mu is None:
            mu = problem.get("mu", None)
        Sigma = problem.get("Σ", None)
        if Sigma is None:
            Sigma = problem.get("Sigma", None)
        gamma = problem.get("γ", None)
        if gamma is None:
            gamma = problem.get("gamma", None)

        if mu is None or Sigma is None or gamma is None:
            return None

        mu = np.asarray(mu, dtype=float)
        Sigma = np.asarray(Sigma, dtype=float)
        try:
            gamma = float(gamma)
        except Exception:
            return None

        n = mu.size
        if Sigma.shape != (n, n):
            try:
                Sigma = Sigma.reshape((n, n))
            except Exception:
                return None

        if n == 0:
            return {"w": []}

        # Symmetrize covariance
        Sigma = 0.5 * (Sigma + Sigma.T)

        # Build quadratic term H = 2γ Σ so that objective is 0.5 w^T H w - mu^T w
        H = (2.0 * gamma) * Sigma

        # Handle degenerate case quickly: if H is near-zero, choose argmax mu
        scale_H = float(np.max(np.abs(H))) if H.size else 0.0
        if not np.isfinite(scale_H) or scale_H <= 1e-16:
            w = np.zeros(n, dtype=float)
            w[int(np.argmax(mu))] = 1.0
            return {"w": w.tolist()}

        # Active-set QP solver on the simplex: minimize 0.5 w^T H w - mu^T w
        # subject to sum(w)=1, w >= 0
        # Working set holds indices fixed at w_i = 0
        # Initialize with uniform feasible point
        w = np.full(n, 1.0 / n, dtype=float)
        Bmask = np.zeros(n, dtype=bool)  # active inequality constraints: w_i == 0

        # Numerical parameters
        # Diagonal regularization for robustness in singular/near-singular cases
        tr = float(np.trace(H))
        eps = 1e-12 * (tr / max(n, 1) + 1.0) + 1e-18
        max_iter = 10 * n + 50

        # Helper to solve KKT on free set S: [H_SS  -1; 1^T 0] [w_S; nu] = [mu_S; 1]
        def solve_kkt(S_idx: np.ndarray) -> tuple[np.ndarray, float]:
            m = S_idx.size
            if m == 1:
                j = int(S_idx[0])
                wS = np.array([1.0], dtype=float)
                nu = float(H[j, j] * 1.0 - mu[j])  # gradient value at free var
                return wS, nu
            K = H[np.ix_(S_idx, S_idx)].astype(float, copy=True)
            # Add tiny ridge for numerical stability
            K.flat[:: m + 1] += eps
            one = np.ones(m, dtype=float)
            M = np.empty((m + 1, m + 1), dtype=float)
            M[:m, :m] = K
            M[:m, m] = -one
            M[m, :m] = one
            M[m, m] = 0.0
            b = np.empty(m + 1, dtype=float)
            b[:m] = mu[S_idx]
            b[m] = 1.0
            try:
                sol = np.linalg.solve(M, b)
            except np.linalg.LinAlgError:
                sol, *_ = np.linalg.lstsq(M, b, rcond=None)
            return sol[:m], float(sol[m])

        # Main active-set loop
        for _ in range(max_iter):
            S = np.flatnonzero(~Bmask)
            m = S.size

            if m == 0:
                # All zero cannot satisfy sum=1; fallback to argmax mu
                w = np.zeros(n, dtype=float)
                w[int(np.argmax(mu))] = 1.0
                break

            # Candidate solution on free set
            wS, nu = solve_kkt(S)
            w_candidate = np.zeros(n, dtype=float)
            w_candidate[S] = wS

            # If any free weight is negative, step toward candidate until one hits zero
            if np.any(wS < -1e-14):
                d = w_candidate - w
                # Only entries with d_i < 0 constrain the step to keep w >= 0
                neg = d < 0.0
                if not np.any(neg):
                    # Numerical oddity; accept candidate
                    w = w_candidate
                    continue
                ratios = -w[neg] / d[neg]
                alpha = float(ratios.min()) if ratios.size else 1.0
                if not np.isfinite(alpha):
                    # Guard: add the most negative component to active set
                    Sneg = S[wS < 0.0]
                    if Sneg.size > 0:
                        Bmask[int(Sneg[np.argmin(wS[wS < 0.0])])] = True
                        continue
                    else:
                        # Fallback
                        w = w_candidate
                        break
                alpha = max(0.0, min(1.0, alpha))
                w = w + alpha * d
                # Clamp near-zero and update active set
                w[np.abs(w) < 1e-15] = 0.0
                Bmask = w <= 0.0
                # Ensure sum remains 1 within roundoff
                s = float(w.sum())
                if abs(s - 1.0) > 1e-12:
                    pos = w > 0.0
                    s_pos = float(w[pos].sum())
                    if s_pos > 0.0:
                        w[pos] *= 1.0 / s_pos
                continue

            # Candidate is nonnegative on S; accept and check multipliers of active set
            wc = w_candidate
            # Clean tiny negatives due to solver noise and renormalize
            wc[wc < 0.0] = 0.0
            s = float(wc.sum())
            if s <= 0.0:
                w = np.zeros(n, dtype=float)
                w[int(np.argmax(mu))] = 1.0
                break
            if abs(s - 1.0) > 1e-12:
                wc /= s

            # Compute gradient and Lagrange multipliers
            g = H.dot(wc) - mu
            if m > 0:
                # Stationarity on free set implies g_i = nu for i in S; use average for robustness
                nu = float(np.mean(g[S]))
            y = g - nu  # multipliers for w >= 0
            y[~Bmask] = 0.0

            if np.any(Bmask):
                yB = y[Bmask]
                # If any active constraint has negative multiplier, remove the most negative from active set
                min_idx_rel = int(np.argmin(yB))
                min_y = float(yB[min_idx_rel])
                scale = 1.0 + abs(nu) + float(np.max(np.abs(g)))
                if min_y < -1e-10 * scale:
                    B_indices = np.flatnonzero(Bmask)
                    Bmask[int(B_indices[min_idx_rel])] = False
                    w = wc
                    continue

            # Optimal
            w = wc
            break

        # Post-process: clamp tiny negatives and renormalize
        w[w < 0.0] = 0.0
        s = float(w.sum())
        if s <= 0.0:
            w = np.zeros(n, dtype=float)
            w[int(np.argmax(mu))] = 1.0
        elif abs(s - 1.0) > 1e-10:
            w /= s

        # Final sanity
        if not np.all(np.isfinite(w)):
            return None
        return {"w": w.tolist()}