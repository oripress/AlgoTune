from typing import Any
import numpy as np
from scipy.integrate import tanhsinh
from scipy.special import wright_bessel
class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Compute elementwise integrals of Wright's Bessel function.
        """
        # Convert inputs to NumPy arrays for efficient broadcasting
        a_vals = np.asarray(problem["a"], dtype=float)
        b_vals = np.asarray(problem["b"], dtype=float)
        lower_vals = np.asarray(problem["lower"], dtype=float)
        upper_vals = np.asarray(problem["upper"], dtype=float)

        # Vectorized integration using tanhsinh (fast and accurate)
        res = tanhsinh(wright_bessel, lower_vals, upper_vals, args=(a_vals, b_vals))
        # The reference guarantees success; we trust the integrator here.
        results = res.integral

        return {"result": results.tolist()}