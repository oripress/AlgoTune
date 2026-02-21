from typing import Any
import numpy as np
from scipy.integrate import tanhsinh
from scipy.special import wright_bessel

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        a = np.array(problem["a"])
        b = np.array(problem["b"])
        lower = np.array(problem["lower"])
        upper = np.array(problem["upper"])
        
        # tanhsinh is vectorized over limits and args
        res = tanhsinh(wright_bessel, lower, upper, args=(a, b))
        
        return {"result": res.integral.tolist()}