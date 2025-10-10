from typing import Any
import numpy as np
from scipy.integrate import tanhsinh
from scipy.special import wright_bessel
from concurrent.futures import ThreadPoolExecutor

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Solve integration problem with scipy.integrate.tanhsinh."""
        a = problem["a"]
        b = problem["b"]
        lower = problem["lower"]
        upper = problem["upper"]
        
        # Use vectorized tanhsinh for best performance
        res = tanhsinh(wright_bessel, lower, upper, args=(a, b))
        assert np.all(res.success)
        return {"result": res.integral.tolist()}