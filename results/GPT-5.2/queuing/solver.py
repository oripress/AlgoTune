from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    """
    Fast closed-form solver.

    Key simplifications:
    - w == d for the given M/M/1 expressions, so only s_i = min(w_max, d_max) matters.
    - q constraint implies r_i = mu_i/lambda_i >= r_q_i where q = 1/(r(r-1)).
    - For fixed r, optimal lambda is the minimum feasible:
        lambda = max(lambda_min, 1/(s*(r-1)))
      hence mu = r*lambda.
    - Objective is sum gamma_i * r_i, increasing in r.
      If sum(mu) is too large, we must increase some r to reduce mu.
    - KKT on the "delay-active" branch gives r-1 = sqrt(nu/(s*gamma)).
      This yields a scalar active-set problem (no CVXPY, no bisection).
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _solve_n1(problem: dict[str, Any]) -> dict[str, Any]:
        import math

        w_max = float(problem["w_max"][0])
        d_max = float(problem["d_max"][0])
        q_max = float(problem["q_max"][0])
        a = float(problem["λ_min"][0])
        mu_max = float(problem["μ_max"])
        g = float(problem["γ"][0])

        s = w_max if w_max < d_max else d_max

        # r lower bound from q <= q_max: q = 1/(r(r-1))
        r_q = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 / q_max))

        # switch where lambda_min becomes active (if a>0); beyond it mu increases
        r_sw = math.inf if a <= 0.0 else 1.0 + 1.0 / (s * a)

        # gamma==0: push to r that minimizes mu (helps feasibility, objective unaffected)
        if g <= 0.0:
            if a <= 0.0:
                r = max(r_q, 1.0e12)
            else:
                r = r_sw if r_q < r_sw else r_q
            lam_delay = 1.0 / (s * (r - 1.0))
            lam = lam_delay if lam_delay > a else a
            mu = r * lam
            return {"μ": np.array([mu]), "λ": np.array([lam]), "objective": 0.0}

        # gamma>0: start at minimal r, check budget
        # At r=r_q, choose minimal lambda
        lam_delay_q = 1.0 / (s * (r_q - 1.0))
        if lam_delay_q > a:
            lam_q = lam_delay_q
            mu_q = (1.0 + 1.0 / (r_q - 1.0)) / s
        else:
            lam_q = a
            mu_q = a * r_q

        if mu_q <= mu_max:
            return {
                "μ": np.array([mu_q]),
                "λ": np.array([lam_q]),
                "objective": float(g * r_q),
            }

        # Need to increase r to reduce mu (only possible on delay-active branch)
        # Solve mu(r) = (1 + 1/(r-1))/s <= mu_max  => r = 1 + 1/(s*mu_max - 1)
        if s * mu_max <= 1.0:
            # infeasible; best effort at r_sw or huge (if no switch)
            r = r_sw if math.isfinite(r_sw) else max(r_q, 1.0e12)
        else:
            r_need = 1.0 + 1.0 / (s * mu_max - 1.0)
            r = r_q if r_need < r_q else r_need
            if math.isfinite(r_sw) and r > r_sw:
                # Can't reduce mu enough before hitting the switch -> infeasible; clip.
                r = r_sw

        lam_delay = 1.0 / (s * (r - 1.0))
        lam = lam_delay if lam_delay > a else a
        mu = r * lam
        return {"μ": np.array([mu]), "λ": np.array([lam]), "objective": float(g * r)}

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        # ultra-fast scalar path
        gamma_arr = problem["γ"]
        if isinstance(gamma_arr, list) and len(gamma_arr) == 1:
            return self._solve_n1(problem)

        w_max = np.asarray(problem["w_max"], dtype=np.float64)
        d_max = np.asarray(problem["d_max"], dtype=np.float64)
        q_max = np.asarray(problem["q_max"], dtype=np.float64)
        a = np.asarray(problem["λ_min"], dtype=np.float64)
        mu_max = float(problem["μ_max"])
        gamma = np.asarray(problem["γ"], dtype=np.float64)

        s = np.minimum(w_max, d_max)
        n = gamma.size

        # r lower bound from q constraint: r >= (1 + sqrt(1 + 4/q_max))/2
        r_q = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / q_max))

        # r_switch = 1 + 1/(s*a) (if a==0 => +inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_sw = 1.0 + 1.0 / (s * a)
        r_sw = np.where(a > 0.0, r_sw, np.inf)

        # Precompute minimal (lambda,mu) at r=r_q
        rm1_q = r_q - 1.0
        lam_delay_q = 1.0 / (s * rm1_q)
        use_delay_q = lam_delay_q > a
        lam_q = np.where(use_delay_q, lam_delay_q, a)
        mu_q = np.where(use_delay_q, (1.0 + 1.0 / rm1_q) / s, a * r_q)

        # For finite switch, mu at r_sw equals a + 1/s (continuous).
        base = 1.0 / s
        mu_sw = np.where(np.isfinite(r_sw), a + base, base)

        # Output arrays
        r = np.empty(n, dtype=np.float64)
        lam = np.empty(n, dtype=np.float64)
        mu = np.empty(n, dtype=np.float64)

        # gamma==0 queues: always minimize mu (objective unaffected, helps feasibility)
        mask0 = gamma <= 0.0
        if mask0.any():
            # If a==0, r_sw is inf; pick a huge finite r to approximate the limit mu=1/s.
            r0 = np.where(a[mask0] > 0.0, r_sw[mask0], 1.0e12)
            r0 = np.where(r_q[mask0] < r0, r0, r_q[mask0])
            rm10 = r0 - 1.0
            lam_delay0 = 1.0 / (s[mask0] * rm10)
            lam0 = np.maximum(a[mask0], lam_delay0)
            mu0 = r0 * lam0

            r[mask0] = r0
            lam[mask0] = lam0
            mu[mask0] = mu0

        # gamma>0 queues that cannot be in the interior (r_q >= r_sw): fixed at r_q
        mask_pos = ~mask0
        mask_fix = mask_pos & ~(r_q < r_sw)
        if mask_fix.any():
            r[mask_fix] = r_q[mask_fix]
            lam[mask_fix] = lam_q[mask_fix]
            mu[mask_fix] = mu_q[mask_fix]

        # gamma>0 queues with potential interior: r in [r_q, r_sw]
        mask_var = mask_pos & (r_q < r_sw)
        if not mask_var.any():
            obj = float(np.dot(gamma, r))
            return {"μ": mu, "λ": lam, "objective": obj}

        # Check if budget is already satisfied with r=r_q for all gamma>0 var queues
        mu_other = float(mu[mask0].sum()) + float(mu[mask_fix].sum())
        sum_mu_rq = mu_other + float(mu_q[mask_var].sum())
        if sum_mu_rq <= mu_max:
            r[mask_var] = r_q[mask_var]
            lam[mask_var] = lam_q[mask_var]
            mu[mask_var] = mu_q[mask_var]
            obj = float(np.dot(gamma, r))
            return {"μ": mu, "λ": lam, "objective": obj}

        # Active-set solve for x = sqrt(nu)
        sv = s[mask_var]
        gv = gamma[mask_var]
        rqv = r_q[mask_var]
        rswv = r_sw[mask_var]
        av = a[mask_var]
        muqv = mu_q[mask_var]
        lamqv = lam_q[mask_var]
        basev = base[mask_var]
        cv = np.sqrt(gv * basev)  # sqrt(gamma/s)

        sqrt_sg = np.sqrt(sv * gv)
        xlow = (rqv - 1.0) * sqrt_sg
        xhigh = np.where(np.isfinite(rswv), (rswv - 1.0) * sqrt_sg, np.inf)
        muup = np.where(np.isfinite(rswv), av + basev, basev)

        low = np.zeros_like(cv, dtype=bool)
        up = np.zeros_like(cv, dtype=bool)
        free = np.ones_like(cv, dtype=bool)

        x = 0.0
        for _ in range(30):
            C = float(cv[free].sum())
            K = (
                mu_other
                + float(muqv[low].sum())
                + float(muup[up].sum())
                + float(basev[free].sum())
            )
            denom = mu_max - K
            if denom <= 0.0:
                x_new = np.inf
            else:
                x_new = C / denom if C > 0.0 else np.inf

            new_low = x_new <= xlow
            new_up = np.isfinite(xhigh) & (x_new >= xhigh)
            new_free = ~(new_low | new_up)

            if (
                x_new == x
                and np.array_equal(new_low, low)
                and np.array_equal(new_up, up)
            ):
                break

            x = x_new
            low, up, free = new_low, new_up, new_free

        # Fill solution for var queues
        idx = np.flatnonzero(mask_var)

        if low.any():
            sel = idx[low]
            r[sel] = rqv[low]
            lam[sel] = lamqv[low]
            mu[sel] = muqv[low]
        if up.any():
            sel = idx[up]
            r[sel] = rswv[up]
            lam[sel] = av[up]
            mu[sel] = muup[up]
        if free.any():
            sel = idx[free]
            # r = 1 + x/sqrt(s*gamma), lambda = c/x, mu = 1/s + c/x
            inv = 1.0 / sqrt_sg[free]
            r_free = 1.0 + x * inv
            lam_free = cv[free] / x
            mu_free = basev[free] + lam_free

            r[sel] = r_free
            lam[sel] = lam_free
            mu[sel] = mu_free

        # Safety: enforce budget if tiny numerical overshoot
        s_mu = float(mu.sum())
        if s_mu > mu_max:
            # Increase free x slightly (reduces mu on free set).
            if free.any() and np.isfinite(x):
                x2 = x * 1.0000000001
                sel = idx[free]
                inv = 1.0 / sqrt_sg[free]
                r_free = 1.0 + x2 * inv
                lam_free = cv[free] / x2
                mu_free = basev[free] + lam_free
                r[sel] = r_free
                lam[sel] = lam_free
                mu[sel] = mu_free

        obj = float(np.dot(gamma, r))
        return {"μ": mu, "λ": lam, "objective": obj}