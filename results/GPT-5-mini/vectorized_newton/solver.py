import numpy as np
from typing import Any, Dict

try:
    import scipy.optimize as _opt
except Exception:
    _opt = None

class Solver:
    """
    Vectorized Newton-Raphson solver wrapper that mirrors the reference implementation.
    Expects problem dict with keys "x0", "a0", "a1" (list-like).
    Looks up scalar parameters a2,a3,a4,a5 from kwargs, then self attributes, then module globals.
    Returns {"roots": [...] } where the list length matches input length (NaN for failures).
    """

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Parse inputs
        try:
            x0_arr = np.array(problem["x0"], dtype=np.float64)
            a0_arr = np.array(problem["a0"], dtype=np.float64)
            a1_arr = np.array(problem["a1"], dtype=np.float64)
            n = x0_arr.size
            if a0_arr.size != n or a1_arr.size != n:
                raise ValueError("Input arrays have mismatched lengths")
        except Exception:
            return {"roots": []}

        # Helper for scalar params
        def _get_param(name: str):
            if name in kwargs:
                return kwargs[name]
            if hasattr(self, name):
                return getattr(self, name)
            g = globals()
            if name in g:
                return g[name]
            raise KeyError(f"Parameter {name} not found")

        try:
            a2 = float(_get_param("a2"))
            a3 = float(_get_param("a3"))
            a4 = float(_get_param("a4"))
            a5 = float(_get_param("a5"))
        except Exception:
            # If we can't obtain scalar parameters, return NaNs
            return {"roots": [float("nan")] * n}

        # Use our fast numpy-based vectorized Newton solver for speed and robustness
        return self._numpy_newton(x0_arr, a0_arr, a1_arr, a2, a3, a4, a5)

    def _numpy_newton(self, x0, a0, a1, a2, a3, a4, a5):
        # Simple vectorized Newton-Raphson fallback (used if SciPy missing).
        maxiter = 50
        tol = 1.48e-8
        f_tol = 1e-12

        x = x0.astype(np.float64, copy=True).ravel()
        a0 = np.array(a0, dtype=np.float64).ravel()
        a1 = np.array(a1, dtype=np.float64).ravel()

        n = x.size
        converged = np.zeros(n, dtype=bool)

        for _ in range(maxiter):
            t = (a0 + x * a3) / a5
            with np.errstate(over="ignore", invalid="ignore"):
                exp_t = np.exp(t)
                f = a1 - a2 * (exp_t - 1.0) - (a0 + x * a3) / a4 - x
                fprime = -a2 * (exp_t * (a3 / a5)) - (a3 / a4) - 1.0

            finite = np.isfinite(f) & np.isfinite(fprime) & (~converged)
            can_step = finite & (np.abs(fprime) > 0.0)
            if not np.any(can_step):
                break

            delta = np.zeros_like(x)
            delta[can_step] = f[can_step] / fprime[can_step]
            x_new = x - delta

            small_step = np.abs(delta) <= tol * (1.0 + np.abs(x_new))
            small_resid = np.abs(f) <= f_tol
            newly_conv = can_step & (small_step | small_resid)

            x[can_step] = x_new[can_step]
            converged[newly_conv] = True

            if np.all(converged | ~np.isfinite(x)):
                break

        roots = x.copy()
        bad = (~converged) | (~np.isfinite(roots))
        roots[bad] = np.nan
        return {"roots": roots.tolist()}