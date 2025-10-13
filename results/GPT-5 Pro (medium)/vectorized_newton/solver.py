from typing import Any, Dict, List

import numpy as np

try:
    from scipy import optimize as _opt
except Exception:  # pragma: no cover
    _opt = None

class Solver:
    """
    Vectorized Newton-Raphson solver for:
      f(x, a0..a5) = a1 - a2*(exp((a0 + x*a3)/a5) - 1) - (a0 + x*a3)/a4 - x

    Expects fixed parameters a2, a3, a4, a5 as attributes on the instance,
    but allows overriding via kwargs in solve(...).
    """

    @staticmethod
    def func(
        x: np.ndarray,
        a0: np.ndarray,
        a1: np.ndarray,
        a2: float,
        a3: float,
        a4: float,
        a5: float,
    ) -> np.ndarray:
        y = (a0 + x * a3) / a5
        return a1 - a2 * (np.exp(y) - 1.0) - (a0 + x * a3) / a4 - x

    @staticmethod
    def fprime(
        x: np.ndarray,
        a0: np.ndarray,
        a1: np.ndarray,
        a2: float,
        a3: float,
        a4: float,
        a5: float,
    ) -> np.ndarray:
        y = (a0 + x * a3) / a5
        return -(a2 * np.exp(y) * (a3 / a5)) - (a3 / a4) - 1.0

    def solve(self, problem: Dict[str, List[float]], **kwargs) -> Dict[str, Any]:
        """
        Finds roots using a vectorized call mirroring scipy.optimize.newton behavior.

        :param problem: Dict with lists "x0", "a0", "a1".
        :return: Dictionary with key "roots": List/np.ndarray of `n` roots. NaN on failure.
        """
        # Parse inputs robustly
        try:
            x0_arr = np.asarray(problem["x0"], dtype=float)
            a0_arr = np.asarray(problem["a0"], dtype=float)
            a1_arr = np.asarray(problem["a1"], dtype=float)
            n = x0_arr.shape[0]
            if a0_arr.shape[0] != n or a1_arr.shape[0] != n:
                return {"roots": []}
        except Exception:
            return {"roots": []}

        # Retrieve fixed parameters; prefer kwargs override, then instance attributes
        try:
            a2 = float(kwargs.get("a2", getattr(self, "a2")))
            a3 = float(kwargs.get("a3", getattr(self, "a3")))
            a4 = float(kwargs.get("a4", getattr(self, "a4")))
            a5 = float(kwargs.get("a5", getattr(self, "a5")))
        except Exception:
            # If parameters are unavailable, cannot proceed meaningfully
            return {"roots": [float("nan")] * n}

        args = (a0_arr, a1_arr, a2, a3, a4, a5)

        # Try SciPy's vectorized newton to match reference closely; fallback to custom Newton if SciPy missing
        try:
            if _opt is not None:
                roots_arr = _opt.newton(self.func, x0_arr, fprime=self.fprime, args=args)
            else:
                # Simple vectorized Newton fallback (kept consistent with SciPy defaults as much as possible)
                x = x0_arr.copy()
                maxiter = 50
                for _ in range(maxiter):
                    f = self.func(x, *args)
                    fp = self.fprime(x, *args)
                    step = f / fp
                    x_new = x - step
                    # Convergence similar to SciPy: step size
                    if np.all(np.abs(step) <= (1.48e-8 + 1.48e-8 * np.abs(x_new))):
                        x = x_new
                        break
                    x = x_new
                else:
                    # Match reference failure behavior
                    raise RuntimeError("Failed to converge")
                roots_arr = x

            roots_out = roots_arr
            if np.isscalar(roots_out):
                roots_out = np.array([roots_out], dtype=float)

        except RuntimeError:
            roots_out = [float("nan")] * n
        except Exception:
            roots_out = [float("nan")] * n

        return {"roots": roots_out}