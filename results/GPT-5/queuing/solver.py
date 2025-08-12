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
        mu_max = float(problem["μ_max"])
        gamma = np.asarray(problem["γ"], dtype=float)

        n = gamma.size
        if not (
            w_max.shape == d_max.shape == q_max.shape == lam_min.shape == gamma.shape == (n,)
        ):
            raise ValueError("All vector inputs must have the same length.")

        # Constants and helpers
        eps = 1e-16  # tighter epsilon to avoid perturbing near-1 optima

        # Effective delay cap tau_i = min(w_max_i, d_max_i), with b_i = 1/tau_i
        tau = np.minimum(w_max, d_max)
        if np.any(tau <= 0):
            raise ValueError("w_max and d_max must be positive.")
        b = 1.0 / tau  # δ_i

        a = lam_min  # a_i

        # Upper bound on rho by q_max: solve rho^2/(1-rho) <= q  => rho <= r(q)
        q = np.maximum(q_max, 0.0)
        # r(q) = (-q + sqrt(q^2 + 4q)) / 2, stabilized as 0.5 * (sqrt(q*(q+4)) - q)
        rho_bar = 0.5 * (np.sqrt(q * (q + 4.0)) - q)
        # Enforce strict stability rho < 1 via rho_bar (already < 1 for finite q)
        rho_bar = np.minimum(rho_bar, 1.0 - 1e-18)
        rho_bar = np.maximum(rho_bar, 0.0)

        # Threshold where a/ρ == b/(1-ρ): θ = a / (a + b)
        denom = a + b
        # denom>0 because b>0 (since tau>0). If a=0 it's fine.
        theta = np.divide(a, denom, out=np.zeros_like(a), where=denom > 0)

        sqrt_gamma = np.sqrt(np.maximum(gamma, 0.0))
        sqrt_b = np.sqrt(b)

        def rho_from_nu(nu: float) -> np.ndarray:
            # ρ* = sqrt(γ) / (sqrt(γ) + sqrt(nu*b))
            if nu < 0:
                nu = 0.0
            s = np.sqrt(nu) * sqrt_b
            denom_r = sqrt_gamma + s
            # If gamma==0, rho_star becomes 0; we'll clamp by theta below
            rho_star = np.divide(sqrt_gamma, denom_r, out=np.zeros_like(sqrt_gamma), where=denom_r > 0)
            # Project to feasible interval [theta, rho_bar]
            rho = np.minimum(rho_bar, np.maximum(theta, rho_star))
            # Ensure strictly positive (avoid division by zero); do not force away from 1
            rho = np.maximum(rho, eps)
            return rho

        def mu_from_rho(rho: np.ndarray) -> np.ndarray:
            # μ = max(a/ρ, b/(1-ρ))
            inv_rho = 1.0 / np.maximum(rho, eps)
            inv_one_minus_rho = 1.0 / np.maximum(1.0 - rho, eps)
            mu_a = a * inv_rho
            mu_b = b * inv_one_minus_rho
            return np.maximum(mu_a, mu_b)

        # Check if at rho = rho_bar we already satisfy the resource constraint (slack case)
        rho0 = np.maximum(rho_bar, eps)
        mu0 = mu_from_rho(rho0)
        sum_mu0 = float(mu0.sum())
        if sum_mu0 <= mu_max + 1e-12:
            rho = rho0
            mu = mu0
            lam = rho * mu
            objective = float(np.dot(gamma, 1.0 / rho))
            return {"μ": mu, "λ": lam, "objective": objective}

        # Compute minimal attainable sum μ at nu -> ∞
        # rho_inf = min(rho_bar, theta)
        rho_inf = np.minimum(rho_bar, theta)
        rho_inf = np.maximum(rho_inf, eps)  # only lower clip
        mu_inf = mu_from_rho(rho_inf)
        sum_mu_inf = float(mu_inf.sum())
        if sum_mu_inf > mu_max + 1e-10:
            # Infeasible problem
            raise ValueError("Solver failed: infeasible constraints.")

        # Find nu such that sum(mu(nu)) ~= mu_max by bisection on nu >= 0
        nu_lo = 0.0
        nu_hi = 1.0
        # Increase nu_hi until resource satisfied
        for _ in range(100):
            rho_hi = rho_from_nu(nu_hi)
            mu_hi = mu_from_rho(rho_hi)
            if float(mu_hi.sum()) <= mu_max:
                break
            nu_hi *= 2.0
        else:
            nu_hi = 1e16

        # Bisection
        rho_best = rho0
        mu_best = mu0
        for _ in range(70):
            nu_mid = 0.5 * (nu_lo + nu_hi)
            rho_mid = rho_from_nu(nu_mid)
            mu_mid = mu_from_rho(rho_mid)
            s_mid = float(mu_mid.sum())
            if s_mid <= mu_max:
                nu_hi = nu_mid
                rho_best = rho_mid
                mu_best = mu_mid
            else:
                nu_lo = nu_mid

        rho = rho_best
        mu = mu_best
        lam = rho * mu

        # Final safety clipping for numerical cleanliness
        rho = np.maximum(rho, eps)
        mu = np.maximum(mu, eps)
        objective = float(np.dot(gamma, 1.0 / rho))
        return {"μ": mu, "λ": lam, "objective": objective}