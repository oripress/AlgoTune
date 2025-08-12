from typing import Any
import numpy as np
from scipy.integrate import tanhsinh
from scipy.special import gamma
import math
from numba import jit

try:
    from scipy.special import wright_bessel
    HAS_WRIGHT_BESSEL = True
except ImportError:
    HAS_WRIGHT_BESSEL = False

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Solve integration problem with numerical integration.

        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        if HAS_WRIGHT_BESSEL:
            # Use the built-in function if available
            a = problem["a"]
            b = problem["b"]
            lower = problem["lower"]
            upper = problem["upper"]
            res = tanhsinh(wright_bessel, lower, upper, args=(a, b))
            # This assert is a sanity check. The goal was to create problems where
            # the integrator always converges using the default tolerances. If this
            # ever fails, then `generate_problem` should be fixed.
            assert np.all(res.success)
            return {"result": res.integral.tolist()}
        else:
            # Fallback implementation
            def wright_bessel_func(x, a_val, b_val):
                """Compute Wright's Bessel function Phi(a, b; x) using series expansion"""
                # Series definition: sum_{k=0}^inf x^k / (k! * Gamma(a*k + b))
                if np.isscalar(x):
                    return self._wright_bessel_scalar(x, a_val, b_val)
                else:
                    # Process array elements
                    try:
                        result = np.zeros_like(x, dtype=float)
                        for i in range(len(x)):
                            result[i] = self._wright_bessel_scalar(x[i], a_val, b_val)
                        return result
                    except TypeError:
                        # x is not array-like, treat as scalar
                        return self._wright_bessel_scalar(x, a_val, b_val)

            a = problem["a"]
            b = problem["b"]
            lower = problem["lower"]
            upper = problem["upper"]

            results = []
            for i in range(len(a)):
                # Use tanh-sinh integration for each parameter set
                def integrand(x):
                    return wright_bessel_func(x, a[i], b[i])

                res = tanhsinh(integrand, lower[i], upper[i])
                results.append(res.integral)

            return {"result": results}

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _wright_bessel_scalar_numba(x, a_val, b_val):
        """Compute Wright's Bessel function Phi(a, b; x) for scalar x using Numba JIT"""
        # Series definition: sum_{k=0}^inf x^k / (k! * Gamma(a*k + b))
        result = 0.0
    def _wright_bessel_scalar(self, x, a_val, b_val):
        """Compute Wright's Bessel function Phi(a, b; x) for scalar x"""
        # Series definition: sum_{k=0}^inf x^k / (k! * Gamma(a*k + b))
        result = 0.0
        k = 0

        # Precompute some frequently used values
        x_power_k = 1.0  # x^k
        factorial_k = 1.0  # k!
        
        # Precompute gamma values for better performance
        # We'll compute gamma values in batches to reduce function calls
        gamma_cache = {}
        
        while True:
            if k == 0:
                gamma_b = gamma_cache.get(b_val)
                if gamma_b is None:
                    gamma_b = gamma(b_val)
                    gamma_cache[b_val] = gamma_b
                term = 1.0 / gamma_b
            else:
                # Update values incrementally to avoid recomputation
                x_power_k *= x
                factorial_k *= k
                
                # Compute gamma(a*k + b) with caching
                gamma_arg = a_val * k + b_val
                gamma_ak_b = gamma_cache.get(gamma_arg)
                if gamma_ak_b is None:
                    gamma_ak_b = gamma(gamma_arg)
                    gamma_cache[gamma_arg] = gamma_ak_b
                term = x_power_k / (factorial_k * gamma_ak_b)

            result += term

            # Check for convergence using relative error
            if abs(term) < 1e-16 * (1 + abs(result)) or k > 150:
                break

            k += 1

        return result