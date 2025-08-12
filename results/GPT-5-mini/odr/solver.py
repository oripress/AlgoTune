from typing import Any, Dict
import numpy as np
import scipy.odr as odr

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Fit weighted orthogonal distance regression (ODR) line.

        Expects problem to have keys "x", "y", "sx", "sy" as lists of numbers.
        Returns {"beta": [slope, intercept]} matching the reference implementation.
        """
        x = np.asarray(problem["x"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        sx = np.asarray(problem["sx"], dtype=float)
        sy = np.asarray(problem["sy"], dtype=float)

        # Avoid zero uncertainties which can cause issues in ODR
        tiny = np.finfo(float).tiny
        sx = np.maximum(sx, tiny)
        sy = np.maximum(sy, tiny)

        data = odr.RealData(x, y=y, sx=sx, sy=sy)
        model = odr.Model(lambda B, x: B[0] * x + B[1])
        output = odr.ODR(data, model, beta0=[0.0, 1.0]).run()
        beta = output.beta.tolist()

        # Safety fallback: if result is not finite, use ordinary least squares.
        if not np.all(np.isfinite(np.asarray(beta, dtype=float))):
            coeffs = np.polyfit(x, y, 1)
            beta = [float(coeffs[0]), float(coeffs[1])]

        return {"beta": beta}