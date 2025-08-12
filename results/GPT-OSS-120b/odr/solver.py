from typing import Any
import numpy as np
import scipy.odr as odr

def _linear_model(B, x):
    """Linear model y = B[0] * x + B[1]."""
    return B[0] * x + B[1]

# Pre‑create the ODR model once to avoid repeated construction overhead
_odr_model = odr.Model(_linear_model)

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Fit a weighted orthogonal distance regression (ODR) line.

        Parameters
        ----------
        problem : dict
            Dictionary with keys "x", "y", "sx", "sy" containing lists of measurements
            and their standard errors.

        Returns
        -------
        dict
            {"beta": [slope, intercept]}
        """
        # Convert inputs to NumPy arrays
        x = np.asarray(problem["x"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        sx = np.asarray(problem["sx"], dtype=float)
        sy = np.asarray(problem["sy"], dtype=float)

        # Use the same initial guess as the reference implementation
        beta0 = [0.0, 1.0]
        
        # Run ODR with default settings (fast convergence)
        data = odr.RealData(x, y=y, sx=sx, sy=sy)
        model = _odr_model
        odr_obj = odr.ODR(data, model, beta0=beta0)
        out = odr_obj.run()
        return {"beta": out.beta.tolist()}