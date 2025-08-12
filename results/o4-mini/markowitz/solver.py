import numpy as np
import ecos
from scipy import sparse

class Solver:
    def solve(self, problem, **kwargs):
        # Load data
        mu = np.asarray(problem["μ"], dtype=float)
        Sigma = np.asarray(problem["Σ"], dtype=float)
        gamma = float(problem["γ"])
        n = mu.size

        # Quadratic term
        H = 2.0 * gamma * Sigma
        tol = 1e-10
        ones = np.ones(n, dtype=float)

        # 1) Try unconstrained KKT solution via block elimination
        try:
            L = np.linalg.cholesky(H)
            y = np.linalg.solve(L, mu)
            z_mu = np.linalg.solve(L.T, y)
            y1 = np.linalg.solve(L, ones)
            z1 = np.linalg.solve(L.T, y1)
            lam = (z_mu.sum() - 1.0) / z1.sum()
            w = z_mu - lam * z1
            if w.min() >= -tol:
                w = np.clip(w, 0.0, None)
                tot = w.sum()
                if tot > 0.0:
                    w /= tot
                else:
                    w.fill(1.0 / n)
                return {"w": w.tolist()}
        except np.linalg.LinAlgError:
            pass

        # 2) Active-set with vectorized elimination
        Z = np.zeros(n, dtype=bool)
        for _ in range(2 * n):
            F = ~Z
            if not np.any(F):
                w = np.ones(n, dtype=float) / n
                return {"w": w.tolist()}

            muF = mu[F]
            HF = H[np.ix_(F, F)]
            try:
                LF = np.linalg.cholesky(HF)
            except np.linalg.LinAlgError:
                # Fallback proportional to positive returns
                w = np.clip(mu, 0.0, None)
                s = w.sum()
                if s <= 0.0:
                    w.fill(1.0 / n)
                else:
                    w /= s
                return {"w": w.tolist()}

            # Solve for free variables
            yF = np.linalg.solve(LF, muF)
            z_muF = np.linalg.solve(LF.T, yF)
            y1F = np.linalg.solve(LF, ones[F])
            z1F = np.linalg.solve(LF.T, y1F)

            lamF = (z_muF.sum() - 1.0) / z1F.sum()
            wF = z_muF - lamF * z1F

            # Primal feasibility: non-negativity
            if np.all(wF >= -tol):
                # Build full weight vector
                w = np.zeros(n, dtype=float)
                w[F] = wF
                # Dual feasibility: check multipliers for active constraints
                if Z.any():
                    nu = H.dot(w) - mu + lamF
                    viol = Z & (nu < -tol)
                    if np.any(viol):
                        Z[viol] = False
                        continue
                # KKT satisfied
                w = np.clip(w, 0.0, None)
                tot = w.sum()
                if tot > 0.0:
                    w /= tot
                else:
                    w.fill(1.0 / n)
                return {"w": w.tolist()}

            # Add all violated constraints at once
            negF = (wF < -tol)
            idxF = np.nonzero(F)[0]
            Z[idxF[negF]] = True

        # Final fallback if loop did not return
        w = np.clip(mu, 0.0, None)
        s = w.sum()
        if s <= 0.0:
            w = np.ones(n, dtype=float) / n
        else:
            w /= s
        return {"w": w.tolist()}