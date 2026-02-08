import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        w_max = np.asarray(problem["w_max"], dtype=float)
        d_max = np.asarray(problem["d_max"], dtype=float)
        q_max = np.asarray(problem["q_max"], dtype=float)
        lam_min = np.asarray(problem["λ_min"], dtype=float)
        mu_max = float(problem["μ_max"])
        gamma = np.asarray(problem["γ"], dtype=float)
        n = gamma.size

        # Key insight: w = q/λ + 1/μ = 1/(μ-λ) = d
        # So w and d constraints both reduce to: μ-λ >= max(1/w_max, 1/d_max)
        delta_min = np.maximum(1.0 / w_max, 1.0 / d_max)

        # q constraint: ρ²/(1-ρ) <= q_max => ρ <= rho_ub
        rho_ub = (-q_max + np.sqrt(q_max * (q_max + 4.0))) / 2.0

        # Crossover: lam_min/ρ = delta_min/(1-ρ) => ρ = lam_min/(lam_min+delta_min)
        rho_cross = lam_min / (lam_min + delta_min)

        # Lower bound on ρ
        rho_lo = np.minimum(rho_cross, rho_ub)
        rho_lo = np.maximum(rho_lo, 1e-12)

        pos = gamma > 0
        n_pos = int(np.sum(pos))

        # Precompute delta_min/gamma ratio for positive gamma queues
        d_over_g = np.zeros(n)
        if n_pos > 0:
            d_over_g[pos] = delta_min[pos] / gamma[pos]

        def compute_rho(nu):
            rho = rho_lo.copy()
            if n_pos > 0:
                if nu <= 0:
                    rho[pos] = rho_ub[pos]
                else:
                    s = np.sqrt(nu * d_over_g[pos])
                    target = 1.0 / (1.0 + s)
                    rho[pos] = np.clip(target, rho_lo[pos], rho_ub[pos])
            return rho

        def total_mu(rho):
            return np.sum(np.maximum(lam_min / rho, delta_min / (1.0 - rho)))

        # Check if ν=0 (all at max ρ) is feasible
        rho_best = compute_rho(0.0)
        if total_mu(rho_best) <= mu_max:
            rho_opt = rho_best
        else:
            # Bisection on Lagrange multiplier ν
            nu_lo_val, nu_hi_val = 0.0, 1.0
            for _ in range(100):
                if total_mu(compute_rho(nu_hi_val)) <= mu_max:
                    break
                nu_hi_val *= 2.0

            for _ in range(100):
                nu_mid = (nu_lo_val + nu_hi_val) * 0.5
                if total_mu(compute_rho(nu_mid)) > mu_max:
                    nu_lo_val = nu_mid
                else:
                    nu_hi_val = nu_mid

            rho_opt = compute_rho((nu_lo_val + nu_hi_val) * 0.5)

        # Recover λ and μ
        lam_opt = np.maximum(lam_min, delta_min * rho_opt / (1.0 - rho_opt))
        mu_opt = lam_opt / rho_opt

        return {"μ": mu_opt, "λ": lam_opt}