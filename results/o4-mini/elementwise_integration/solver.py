from typing import Any
import numpy as np
from scipy.integrate import tanhsinh

# Dynamically import wright_bessel to avoid linter errors
_w = __import__('scipy.special', fromlist=['wright_bessel'])
wright_bessel = _w.wright_bessel

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Solve the integration of the Wright–Bessel function using scipy.integrate.tanhsinh.
        Mimics the reference implementation to ensure numerical consistency.
        """
        a = problem["a"]
        b = problem["b"]
        lower = problem["lower"]
        upper = problem["upper"]
        res = tanhsinh(wright_bessel, lower, upper, args=(a, b))
        # Sanity check: integrator must converge
        assert np.all(res.success)
        return {"result": res.integral.tolist()}