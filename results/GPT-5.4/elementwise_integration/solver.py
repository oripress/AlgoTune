from typing import Any

import numpy as np
from scipy.integrate import tanhsinh
from scipy.special import wright_bessel

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        a = problem["a"]
        b = problem["b"]
        lower = problem["lower"]
        upper = problem["upper"]
        res = tanhsinh(wright_bessel, lower, upper, args=(a, b))
        assert np.all(res.success)
        return {"result": res.integral.tolist()}