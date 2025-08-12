from typing import Any, Dict
import numpy as np
from scipy.integrate import tanhsinh
from scipy.special import wright_bessel

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Solve integration problem with scipy.integrate.tanhsinh."""
        a = np.asarray(problem["a"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        lower = np.asarray(problem["lower"], dtype=np.float64)
        upper = np.asarray(problem["upper"], dtype=np.float64)

        res = tanhsinh(wright_bessel, lower, upper, args=(a, b))
        assert np.all(res.success)
        return {"result": res.integral.tolist()}