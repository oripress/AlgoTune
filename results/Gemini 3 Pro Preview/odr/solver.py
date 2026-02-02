from typing import Any
import numpy as np
import scipy.odr as odr

def fcn(B, x):
    return B[0]*x + B[1]

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        x = np.asarray(problem["x"])
        y = np.asarray(problem["y"])
        sx = np.asarray(problem["sx"])
        sy = np.asarray(problem["sy"])
        
        wd = 1.0 / (sx**2)
        we = 1.0 / (sy**2)
        
        output = odr.odr(fcn, [0.0, 1.0], y, x, we=we, wd=wd)
        
        return {"beta": output[0].tolist()}