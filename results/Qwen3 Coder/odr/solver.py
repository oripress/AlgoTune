import numpy as np
from typing import Any
import scipy.odr as odr

class Solver:
    def __init__(self):
        # Pre-create the model to avoid repeated lambda creation
        self.model = odr.Model(self._linear_func)
    
    @staticmethod
    def _linear_func(B, x):
        return B[0] * x + B[1]
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Fit weighted ODR with scipy.odr."""
        x = np.asarray(problem["x"])
        y = np.asarray(problem["y"])
        sx = np.asarray(problem["sx"])
        sy = np.asarray(problem["sy"])

        data = odr.RealData(x, y=y, sx=sx, sy=sy)
        data = odr.RealData(x, y=y, sx=sx, sy=sy)
        odr_obj = odr.ODR(data, self.model, beta0=[0.0, 1.0], maxit=15)
        odr_obj.set_iprint(0)  # Suppress output
        output = odr_obj.run()
        return {"beta": output.beta.tolist()}