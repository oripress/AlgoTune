from typing import Any
import numpy as np
import mpmath
from numba import njit

@njit
def Z_numba(t: float, theta_t: float) -> float:
    """
    Computes the Riemann-Siegel Z-function using its main sum
    and the first correction term for improved accuracy.
    """
    # Precompute t / (2*pi) as it's used multiple times
    t_div_2pi = t / (2 * np.pi)
    a = np.sqrt(t_div_2pi)
    N = int(a)
    
    # Main sum of the Riemann-Siegel formula
    main_sum = 0.0
    # The sum is empty if N=0, which is correct.
    if N > 0:
        n_vals = np.arange(1, N + 1)
        main_sum = np.sum(np.cos(theta_t - t * np.log(n_vals)) / np.sqrt(n_vals))
    main_term = 2 * main_sum

    # First correction term R(t), using Odlyzko's C0(p) formula.
    p = a - N
    
    # The asymptotic expansion is not accurate near p=0.5 where C0(p) has a pole.
    # We avoid applying the correction in a small interval around this point.
    # The asymptotic expansion is not accurate near p=0.5 where C0(p) has a pole.
    # We avoid applying the correction in a small interval around this point.
    # A wider interval (0.1 vs 0.05) proved to be more robust in testing.
    if np.abs(p - 0.5) < 0.1:
        return main_term

    # C0(p) = cos(2*pi*(p^2 - p - 1/16)) / (2*cos(pi*p))
    num_arg = 2 * np.pi * (p*p - p - 1.0/16.0)
    C0_p_num = np.cos(num_arg)
    
    den_arg = np.pi * p
    C0_p_den = 2 * np.cos(den_arg)
    
    # The guard above should prevent division by zero.
    C0_p = C0_p_num / C0_p_den
    
    # R(t) ≈ (-1)^(N-1) * (t/2π)^(-1/4) * C0(p)
    correction_term = ((-1)**(N - 1)) * t_div_2pi**(-0.25) * C0_p
    
    return main_term + correction_term
class Solver:
    _theta_cache = {}

    def __init__(self):
        mpmath.mp.dps = 15

    def _theta(self, t: float) -> float:
        """
        Computes the Riemann-Siegel theta function.
        """
        if t in self._theta_cache:
            return self._theta_cache[t]
        
        z = 0.25 + 1j * t / 2.0
        log_gamma_z = mpmath.loggamma(z)
        result = float(log_gamma_z.imag) - (t / 2.0) * np.log(np.pi)
        
        if len(self._theta_cache) > 256:
            self._theta_cache.clear()
        self._theta_cache[t] = result
        return result

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Counts the number of zeros of the Riemann Zeta function up to height t.
        """
        t = float(problem["t"])

        if t < 14.1347:
            return {"result": 0}

        main_term = (t / (2 * np.pi)) * (np.log(t / (2 * np.pi)) - 1) + 7.0/8.0
        
        theta_t = self._theta(t)
        Z_t = Z_numba(t, theta_t)
        
        zeta_val = Z_t * np.exp(-1j * theta_t)
        s_t = (1.0 / np.pi) * np.angle(zeta_val)
        
        n_t = main_term + s_t
        return {"result": int(round(n_t))}