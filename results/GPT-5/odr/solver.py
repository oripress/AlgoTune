from typing import Any, Dict
import numpy as np
from scipy import odr

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Fit a weighted orthogonal distance regression (linear model).

        Minimizes sum_i [(dx_i / sx_i)^2 + (dy_i / sy_i)^2] subject to y = a x + b.
        Returns slope a and intercept b as in the reference implementation.
        """
        x = np.asarray(problem["x"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        sx = np.asarray(problem["sx"], dtype=float)
        sy = np.asarray(problem["sy"], dtype=float)

        # Ensure finiteness; replace any invalids conservatively
        if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(sx).all() and np.isfinite(sy).all()):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            # Avoid zero or invalid uncertainties; fall back to 1.0 where invalid
            sx = np.nan_to_num(sx, nan=1.0, posinf=1.0, neginf=1.0)
            sy = np.nan_to_num(sy, nan=1.0, posinf=1.0, neginf=1.0)

        data = odr.RealData(x, y=y, sx=sx, sy=sy)
        model = odr.Model(lambda B, x_: B[0] * x_ + B[1])
        output = odr.ODR(data, model, beta0=[0.0, 1.0]).run()

        beta = output.beta
        # Fallback to weighted LS on (x, y) if something went awry
        if not np.all(np.isfinite(beta)):
            w = 1.0 / (sy * sy + 1e-300)
            W = np.sum(w)
            x_bar = np.sum(w * x) / W
            y_bar = np.sum(w * y) / W
            Sxx = np.sum(w * (x - x_bar) ** 2)
            Sxy = np.sum(w * (x - x_bar) * (y - y_bar))
            a = Sxy / Sxx if Sxx > 0 else 0.0
            b = y_bar - a * x_bar
            beta = np.array([a, b], dtype=float)

        return {"beta": beta.tolist()}