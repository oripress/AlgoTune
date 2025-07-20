import numpy as np
from scipy.optimize import newton

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Vectorized Newton-Raphson for f(x) = a1 - a2*(exp((a0 + x*a3)/a5) - 1)
                         - (a0 + x*a3)/a4 - x, for n instances.
        Expects parameters a2, a3, a4, a5 passed via kwargs.
        """
        # Parse inputs
        try:
            x0 = np.asarray(problem["x0"], dtype=float)
            a0 = np.asarray(problem["a0"], dtype=float)
            a1 = np.asarray(problem["a1"], dtype=float)
        except Exception:
            return {"roots": []}

        n = x0.size
        if a0.size != n or a1.size != n:
            return {"roots": []}

        # Extract fixed parameters
        try:
            a2 = float(kwargs["a2"])
            a3 = float(kwargs["a3"])
            a4 = float(kwargs["a4"])
            a5 = float(kwargs["a5"])
        except Exception:
            return {"roots": []}

        # Attempt vectorized Newton-Raphson
        try:
            roots = newton(
                func=self._func,
                x0=x0,
                fprime=self._fprime,
                args=(a0, a1, a2, a3, a4, a5),
                tol=1e-12,
                maxiter=50,
            )
        except RuntimeError:
            return {"roots": [float("nan")] * n}
        except Exception:
            return {"roots": [float("nan")] * n}

        roots = np.atleast_1d(roots)
        if roots.size == 1 and n > 1:
            roots = np.full(n, roots.item(), dtype=float)
        elif roots.size < n:
            roots = np.concatenate([roots, np.full(n - roots.size, np.nan)])
        elif roots.size > n:
            roots = roots[:n]

        return {"roots": roots.tolist()}

    @staticmethod
    def _func(x, a0, a1, a2, a3, a4, a5):
        t = (a0 + x * a3) / a5
        return a1 - a2 * (np.exp(t) - 1.0) - (a0 + x * a3) / a4 - x

    @staticmethod
    def _fprime(x, a0, a1, a2, a3, a4, a5):
        t = (a0 + x * a3) / a5
        return -a2 * np.exp(t) * (a3 / a5) - (a3 / a4) - 1.0