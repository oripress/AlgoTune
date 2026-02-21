from typing import Any
import numpy as np
from scipy.optimize import least_squares

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        x = np.asarray(problem["x"])
        y = np.asarray(problem["y"])
        sx = np.asarray(problem["sx"])
        sy = np.asarray(problem["sy"])
        
        sx2 = sx**2
        sy2 = sy**2
        
        def residuals(B):
            B0, B1 = B
            return (y - B0 * x - B1) / np.sqrt(sy2 + B0**2 * sx2)
            
        # scipy.odr uses beta0=[0.0, 1.0]
        res = least_squares(residuals, x0=[0.0, 1.0], method='lm', xtol=1e-12, ftol=1e-12)
        
        return {"beta": res.x.tolist()}