from __future__ import annotations

from typing import Any, Dict

import numpy as np


class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Parse inputs
        w_max = np.asarray(problem["w_max"], dtype=float)
        d_max = np.asarray(problem["d_max"], dtype=float)
        q_max = np.asarray(problem["q_max"], dtype=float)
        lam_min = np.asarray(problem["λ_min"], dtype=float)
        mu_max_total = float(problem["μ_max"])
        gamma = np.asarray(problem["γ"], dtype=float)

        n = gamma.size
        if not (w_max.size == d_max.size == q_max.size == lam_min.size == n):
            raise ValueError("Input dimensions do not match.")

        # Derived per-queue parameter: delta = max(1/w_max, 1/d_max)
        # This ensures w <= w_max and d <= d_max via mu(1 - rho) >= delta
        inv_w = np.divide(1.0, w_max, out=np.zeros_like(w_max), where=w_max > 0)
        inv_d = np.divide(1.0, d_max, out=np.zeros_like(d_max), where=d_max > 0)
        delta = np.maximum(inv_w, inv_d)

        # Rho upper bound from q_max: solve rho^2 / (1 - rho) <= q  => positive root
        # Stable formula: rho = 2q / (sqrt(q^2 + 4q) + q)
        q = q_max
        # Handle q < 0 (should not happen) by capping at 0
        q = np.maximum(q, 0.0)
        sqrt_term = np.sqrt(q * (q + 4.0))
        # Avoid 0/0 for q=0
        denom = sqrt_term + q
        rho_qmax = np.divide(2.0 * q, denom, out=np.zeros_like(q), where=denom > 0)

        # Enforce strict stability rho < 1 (numerical margin)
        eps = 1e-12
        rho_upper = np.minimum(rho_qmax, 1.0 - eps)

        # Crossing point where lam_min/rho equals delta/(1 - rho)
        # rho_cross = lam_min / (lam_min + delta); if lam_min + delta == 0, set 0
        sum_ld = lam_min + delta
        rho_cross = np.divide(lam_min, sum_ld, out=np.zeros_like(sum_ld), where=sum_ld > 0)

        # Quick infeasibility checks:
        # If rho_upper <= 0 but lam_min > 0, cannot serve any arrival -> infeasible
        if np.any((rho_upper <= 0) & (lam_min > 0)):
            raise ValueError("Infeasible: q_max too tight for positive λ_min.")

        # Partition indices: whether B region (delta term) is reachable
        has_B = rho_upper > rho_cross

        # Minimal achievable mu per queue as t -> ∞
        # If has_B: rho -> rho_cross and mu -> lam_min + delta
        # Else: rho -> rho_upper and mu -> lam_min / rho_upper
        mu_min_per_queue = np.empty(n, dtype=float)
        # Branch: has_B True
        idx_B = has_B
        mu_min_per_queue[idx_B] = lam_min[idx_B] + delta[idx_B]
        # Branch: has_B False
        idx_Aonly = ~idx_B
        # Avoid division by 0 by checking rho_upper > 0; if 0, set very large (infeasible later)
        with np.errstate(divide="ignore"):
            mu_min_per_queue[idx_Aonly] = np.divide(
                lam_min[idx_Aonly],
                rho_upper[idx_Aonly],
                out=np.full(np.count_nonzero(idx_Aonly), np.inf),
                where=rho_upper[idx_Aonly] > 0,
            )

        mu_min_total = float(np.sum(mu_min_per_queue))
        if mu_min_total - mu_max_total > 1e-12:
            # Infeasible problem: even at best (minimal) mu usage, exceeds cap
            raise ValueError("Infeasible: μ_max too small given constraints.")

        # Helper to compute rho, mu, lam for a given dual t (vectorized)
        def compute_for_t(t: float):
            # For t=0: rho = rho_upper (always optimal, monotone decreasing objective)
            if t == 0.0:
                rho = rho_upper.copy()
            else:
                # Unconstrained minimizer in region B: rho_hat = 1 / (1 + sqrt(t * delta / gamma))
                # Handle gamma == 0 specially: take rho_hat -> 0 to minimize t * delta / (1 - rho)
                # For delta == 0, ratio is 0 and rho_hat = 1
                ratio = np.zeros(n, dtype=float)
                # For gamma > 0
                mask_gpos = gamma > 0
                ratio[mask_gpos] = t * delta[mask_gpos] / gamma[mask_gpos]
                # For gamma == 0 and delta > 0 -> set ratio = +inf so rho_hat -> 0
                mask_gzero = ~mask_gpos
                ratio[mask_gzero & (delta > 0)] = np.inf
                # For gamma == 0 and delta == 0 -> 0/0 -> treat as 0 (rho_hat=1)
                ratio[mask_gzero & (delta == 0)] = 0.0

                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    rho_hat = 1.0 / (1.0 + np.sqrt(ratio))

                # Constrained optimum:
                # If rho_upper <= rho_cross: rho = rho_upper (region A only)
                # Else rho = clip(rho_hat, rho_cross, rho_upper)
                rho = rho_upper.copy()
                # For those with available B region, clip
                if np.any(has_B):
                    tmp = rho_hat[has_B]
                    tmp = np.maximum(tmp, rho_cross[has_B])
                    tmp = np.minimum(tmp, rho_upper[has_B])
                    rho[has_B] = tmp

            # Compute mu lower bound per rho: max(lam_min / rho, delta / (1 - rho))
            with np.errstate(divide="ignore", invalid="ignore"):
                term_A = np.divide(lam_min, rho, out=np.full_like(rho, np.inf), where=rho > 0)
                term_B = np.divide(delta, 1.0 - rho, out=np.full_like(rho, np.inf), where=(1.0 - rho) > 0)
            mu = np.maximum(term_A, term_B)

            # Compute lambda = rho * mu
            lam = rho * mu
            mu_sum = float(np.sum(mu))
            return rho, mu, lam, mu_sum

        # First check t=0: if within cap, that's optimal (max rho -> min objective)
        rho0, mu0, lam0, mu_sum0 = compute_for_t(0.0)
        if mu_sum0 <= mu_max_total + 1e-12:
            return {
                "μ": mu0,
                "λ": lam0,
                "objective": float(np.sum(gamma / rho0)),
            }

        # Otherwise, find t > 0 such that sum mu == mu_max_total via bisection
        # Bracket t: start from 1 and grow until mu_sum <= mu_max_total
        t_lo = 0.0
        t_hi = 1.0
        # Cap on iterations for safety
        for _ in range(100):
            _, _, _, mu_sum = compute_for_t(t_hi)
            if mu_sum <= mu_max_total:
                break
            t_hi *= 4.0
        else:
            # If even huge t doesn't reduce enough (shouldn't happen due to mu_min check), raise
            raise ValueError("Failed to bracket dual variable.")

        # Bisection
        # Target precision on mu_sum
        tol = max(1e-10, 1e-12 * max(1.0, mu_max_total))
        for _ in range(80):
            t_mid = 0.5 * (t_lo + t_hi)
            rho_m, mu_m, lam_m, mu_sum_m = compute_for_t(t_mid)
            if mu_sum_m > mu_max_total:
                t_lo = t_mid
            else:
                t_hi = t_mid
            if abs(mu_sum_m - mu_max_total) <= tol:
                break

        # Use upper bound (feasible) solution
        rho, mu, lam, _ = compute_for_t(t_hi)

        # Compute objective
        with np.errstate(divide="ignore"):
            obj_val = float(np.sum(gamma / rho))

        return {"μ": mu, "λ": lam, "objective": obj_val}