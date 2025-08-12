from typing import Any
import numpy as np
from scipy.integrate import tanhsinh
from scipy.special import wright_bessel

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Solve integration problem with tanhsinh for accuracy."""
        a = problem["a"]
        b = problem["b"]
        lower = problem["lower"]
        upper = problem["upper"]
        
        # Use tanhsinh for all integrals to ensure accuracy
        res = tanhsinh(wright_bessel, lower, upper, args=(a, b))
        
        return {"result": res.integral.tolist()}