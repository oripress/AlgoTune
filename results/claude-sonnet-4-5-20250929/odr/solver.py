from typing import Any
import numpy as np
import scipy.odr as odr

class Solver:
    def __init__(self):
        # Pre-define the linear model function
        def linear_func(B, x):
            return B[0] * x + B[1]
        self.model = odr.Model(linear_func)
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Optimized weighted ODR with scipy.odr."""
        # Use views instead of copying when possible
        x = np.asarray(problem["x"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        sx = np.asarray(problem["sx"], dtype=np.float64)
        sy = np.asarray(problem["sy"], dtype=np.float64)
        
        data = odr.RealData(x, y=y, sx=sx, sy=sy)
        output = odr.ODR(data, self.model, beta0=[0.0, 1.0]).run()
        return {"beta": output.beta.tolist()}