from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        w_max = np.asarray(problem["w_max"], dtype=float)
        d_max = np.asarray(problem["d_max"], dtype=float)
        q_max = np.asarray(problem["q_max"], dtype=float)
        lam_min = np.asarray(problem["λ_min"], dtype=float)
        mu_max = float(problem["μ_max"])
        gamma = np.asarray(problem["γ"], dtype=float)

        n = gamma.size
        if (
            w_max.shape != (n,)
            or d_max.shape != (n,)
            or q_max.shape != (n,)
            or lam_min.shape != (n,)
        ):
            raise ValueError("Incompatible input shapes")

        t = np.minimum(w_max, d_max)
        if np.any(t <= 0) or np.any(q_max <= 0) or np.any(lam_min < 0):
            raise ValueError("Infeasible bounds")

        beta = 1.0 / t

        # q = 1 / (ell * (ell - 1)) <= q_max
        ell_q = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / q_max))

        with np.errstate(divide="ignore", invalid="ignore"):
            ell_switch = np.where(lam_min > 0, 1.0 + beta / lam_min, np.inf)

        ell_lb = np.nextafter(ell_q, np.inf)

        ell = np.empty(n, dtype=float)
        lam = np.empty(n, dtype=float)

        fixed = ell_lb >= ell_switch
        active = ~fixed

        if np.any(fixed):
            ell[fixed] = ell_lb[fixed]
            lam[fixed] = np.maximum(lam_min[fixed], beta[fixed] / (ell[fixed] - 1.0))

        y = np.empty(n, dtype=float)
        fixed_service = float(np.sum(ell[fixed] * lam[fixed])) if np.any(fixed) else 0.0

        if np.any(active):
            y_lo = ell_lb[active] - 1.0
            y_hi = np.where(np.isfinite(ell_switch[active]), ell_switch[active] - 1.0, np.inf)

            gam_a = gamma[active]
            beta_a = beta[active]
            lam_min_a = lam_min[active]

            zero = gam_a <= 0
            y_a = np.empty_like(y_lo)

            if np.any(zero):
                y_a[zero] = np.where(
                    np.isfinite(y_hi[zero]), y_hi[zero], np.maximum(y_lo[zero], 1e12)
                )
                fixed_service += float(np.sum(beta_a[zero] * (1.0 + 1.0 / y_a[zero])))

            pos = ~zero
            if np.any(pos):
                b = beta_a[pos]
                g = gam_a[pos]
                a_lo = y_lo[pos]
                a_hi = y_hi[pos]

                service_at_lo = fixed_service + float(np.sum(b * (1.0 + 1.0 / a_lo)))
                if service_at_lo <= mu_max + 1e-12:
                    y_pos = a_lo
                else:
                    inv_hi = np.where(np.isfinite(a_hi), 1.0 / a_hi, 0.0)
                    service_inf = fixed_service + float(np.sum(b * (1.0 + inv_hi)))
                    if service_inf > mu_max + 1e-10:
                        raise ValueError("Solver failed with status infeasible")

                    target = mu_max - fixed_service - float(np.sum(b))

                    def lhs(tau: float) -> float:
                        yy = np.sqrt(tau * b / g)
                        yy = np.maximum(yy, a_lo)
                        yy = np.minimum(yy, a_hi)
                        return float(np.sum(b / yy))

                    lo_tau = 0.0
                    hi_tau = 1.0
                    while lhs(hi_tau) > target:
                        hi_tau *= 4.0

                    for _ in range(80):
                        mid = 0.5 * (lo_tau + hi_tau)
                        if lhs(mid) > target:
                            lo_tau = mid
                        else:
                            hi_tau = mid

                    y_pos = np.sqrt(hi_tau * b / g)
                    y_pos = np.maximum(y_pos, a_lo)
                    y_pos = np.minimum(y_pos, a_hi)

                y_a[pos] = y_pos

            y[active] = y_a
            ell[active] = y_a + 1.0
            lam[active] = np.maximum(lam_min_a, beta_a / y_a)

        mu = ell * lam
        objective = float(np.dot(gamma, ell))
        return {"μ": mu, "λ": lam, "objective": objective}