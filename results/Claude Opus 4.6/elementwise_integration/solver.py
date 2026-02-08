import numpy as np
import numba as nb

@nb.njit(cache=True, fastmath=True)
def _wright_bessel_val(a, b, x):
    """Compute Phi(a, b; x) = sum x^k / (k! * Gamma(a*k + b))"""
    # Use log-space to avoid overflow/underflow
    result = 0.0
    if x <= 0.0:
        # Only k=0 term: 1/Gamma(b)
        return np.exp(-nb.literally(0.0) + (-_lgamma(b)))
    
    log_x = np.log(x)
    log_fact = 0.0  # log(k!)
    
    for k in range(100):
        log_gamma_arg = a * k + b
        log_term = k * log_x - log_fact - _lgamma(log_gamma_arg)
        
        if log_term > -700:
            term = np.exp(log_term)
            result += term
            if k > 3 and term < 1e-15 * result:
                break
        elif k > 5 and log_term < -700:
            break
        
        log_fact += np.log(float(k + 1))
    
    return result

@nb.njit(cache=True, fastmath=True)
def _lgamma(x):
    """Log-gamma using numba-supported math."""
    import math
    return math.lgamma(x)

@nb.njit(cache=True, parallel=True, fastmath=True)
def _integrate_all(a_arr, b_arr, lower_arr, upper_arr, nodes, weights):
    n = len(a_arr)
    m = len(nodes)
    result = np.empty(n)
    
    for i in nb.prange(n):
        mid = (upper_arr[i] + lower_arr[i]) * 0.5
        half = (upper_arr[i] - lower_arr[i]) * 0.5
        s = 0.0
        for j in range(m):
            x_pt = mid + half * nodes[j]
            # tanhsinh calls wright_bessel(x, a, b) -> Phi(x_pt, a[i], b[i]; ...)
            # So first arg = x_pt (-> a param), second = a[i] (-> b param), third = b[i] (-> x param)
            s += weights[j] * _wright_bessel_val(x_pt, a_arr[i], b_arr[i])
        result[i] = half * s
    
    return result

class Solver:
    def __init__(self):
        self.nodes, self.weights = np.polynomial.legendre.leggauss(30)
        # Warm up JIT compilation
        _a = np.array([1.0])
        _b = np.array([1.0])
        _l = np.array([1.0])
        _u = np.array([7.0])
        _integrate_all(_a, _b, _l, _u, self.nodes, self.weights)
    
    def solve(self, problem, **kwargs):
        a = np.asarray(problem["a"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        lower = np.asarray(problem["lower"], dtype=np.float64)
        upper = np.asarray(problem["upper"], dtype=np.float64)
        
        result = _integrate_all(a, b, lower, upper, self.nodes, self.weights)
        
        return {"result": result.tolist()}