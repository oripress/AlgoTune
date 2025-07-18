from typing import Any
import numpy as np
from scipy.special import gammaln

def _gamma_sign(z: np.ndarray) -> np.ndarray:
    """
    Computes the sign of the Gamma function for real arguments.
    This is robust to floating point inaccuracies.
    """
    s = np.ones_like(z, dtype=np.float64)
    
    # Use a tolerance to check for non-integers
    is_integer = np.abs(z - np.round(z)) < 1e-9
    neg_mask = (z < 0) & ~is_integer
    
    if np.any(neg_mask):
        z_neg = z[neg_mask]
        # For z in (-(k+1), -k), sign is (-1)^k.
        # k = floor(-z).
        num_poles_crossed = np.floor(-z_neg)
        s[neg_mask] = (-1.0)**num_poles_crossed
        
    return s

def _compute_indefinite_integral_series(a, b, x):
    """
    Computes the indefinite integral of Wright's Bessel function, evaluated at x,
    using a robust, non-recurrent series summation.
    \Psi(a, b; x) = \sum_{k=0}^{\infty} \frac{x^{k+1}}{(k+1)!\Gamma(ak + b)}
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    total_sum = np.zeros_like(x)

    active_mask = (x != 0)
    if not np.any(active_mask):
        return total_sum

    # Work only with the active elements for efficiency
    active_a, active_b, active_x = a[active_mask], b[active_mask], x[active_mask]
    sign_x = np.sign(active_x)
    active_sum = np.zeros_like(active_x)

    with np.errstate(divide='ignore', invalid='ignore'):
        log_abs_x = np.log(np.abs(active_x))

        max_iterations = 300
        for k in range(max_iterations):
            z = active_a * k + active_b
            
            # log(|T_k|) = log(|x^(k+1)|) - log((k+1)!) - log(|Gamma(ak+b)|)
            log_abs_term = (k + 1) * log_abs_x - gammaln(k + 2) - gammaln(z)

            # sign(T_k) = sign(x^(k+1)) / sign(Gamma(ak+b))
            # which is sign(x^(k+1)) * sign(Gamma(ak+b)) since sign is +/- 1
            sign_x_pow = np.power(sign_x, k + 1)
            sign_g = _gamma_sign(z)
            total_sign = sign_x_pow / sign_g
            
            current_term = total_sign * np.exp(log_abs_term)
            
            # Poles in Gamma or inf-inf lead to NaN; these terms are mathematically zero.
            current_term[np.isnan(current_term)] = 0.0
            
            active_sum += current_term
    
    total_sum[active_mask] = active_sum
    return total_sum

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Computes the definite integral of Wright's Bessel function by evaluating
        the power series of its indefinite integral at the integration bounds.
        """
        a = problem["a"]
        b = problem["b"]
        lower = problem["lower"]
        upper = problem["upper"]

        val_upper = _compute_indefinite_integral_series(a, b, upper)
        val_lower = _compute_indefinite_integral_series(a, b, lower)

        result = val_upper - val_lower
        return {"result": result.tolist()}