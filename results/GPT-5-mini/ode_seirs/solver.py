from typing import Any, Dict
import numpy as np

# Prefer SciPy's integrator for accuracy; fallback if unavailable.
try:
    from scipy.integrate import solve_ivp
except Exception:
    solve_ivp = None

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        Solve the SEIRS ODE system and return [S, E, I, R] at final time t1.
        Uses scipy.integrate.solve_ivp (RK45) with tight tolerances when available.
        """
        if not isinstance(problem, dict):
            raise ValueError("problem must be a dict with keys 't0','t1','y0','params'")

        if not all(k in problem for k in ("t0", "t1", "y0", "params")):
            raise ValueError("problem missing required keys 't0','t1','y0','params'")

        y0 = np.asarray(problem["y0"], dtype=float)
        if y0.size != 4:
            y0 = np.resize(y0, 4).astype(float)

        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem.get("params", {}) or {}

        beta = float(params.get("beta", 0.0))
        sigma = float(params.get("sigma", 0.0))
        gamma = float(params.get("gamma", 0.0))
        omega = float(params.get("omega", 0.0))

        # Trivial case: no time integration required
        if t1 == t0:
            total = float(np.sum(y0))
            if total > 0 and np.isfinite(total):
                return (y0 / total).tolist()
            return y0.tolist()

        # Pre-bind parameters for speed
        b = beta
        sig = sigma
        g = gamma
        w = omega

        def seirs(t, y):
            S, E, I, R = y
            dSdt = -b * S * I + w * R
            dEdt = b * S * I - sig * E
            dIdt = sig * E - g * I
            dRdt = g * I - w * R
            return np.array([dSdt, dEdt, dIdt, dRdt], dtype=float)

        # Primary integrator: SciPy's solve_ivp with RK45 and tight tolerances
        if solve_ivp is not None:
            sol = solve_ivp(seirs, (t0, t1), y0, method="RK45", rtol=1e-10, atol=1e-10)
            if getattr(sol, "success", False):
                return np.asarray(sol.y[:, -1], dtype=float).tolist()
            raise RuntimeError(f"Solver failed: {getattr(sol, 'message', 'solve_ivp failed')}")

        # Fallback: fixed-step RK4 integrator
        T = t1 - t0
        nsteps = int(max(1000, min(200000, int(abs(T) * 1000))))
        if nsteps <= 0:
            nsteps = 1000
        h = T / nsteps

        y = y0.astype(float).copy()
        t = t0
        for _ in range(nsteps):
            k1 = seirs(t, y)
            k2 = seirs(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = seirs(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = seirs(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h

        # Post-process: ensure finite, non-negative, normalized
        y = np.where(np.isfinite(y), y, 0.0)
        y = np.maximum(y, 0.0)
        total = float(np.sum(y))
        if total > 0.0:
            y = y / total
        else:
            total0 = float(np.sum(y0))
            if total0 > 0.0:
                y = y0 / total0
            else:
                y = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        return y.tolist()