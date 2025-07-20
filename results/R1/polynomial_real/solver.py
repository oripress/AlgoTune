import numpy as np
import numba

class Solver:
    def solve(self, problem, **kwargs):
        coeffs = np.array(problem, dtype=np.float64)
        n = len(coeffs) - 1
        if n == 0:
            return []
        if n == 1:
            return [-coeffs[1] / coeffs[0]]
        if n == 2:
            a, b, c = coeffs
            disc = b**2 - 4*a*c
            r1 = (-b + np.sqrt(disc)) / (2*a)
            r2 = (-b - np.sqrt(disc)) / (2*a)
            return sorted([r1, r2], reverse=True)
        
        # Cauchy bound for initial roots
        R = 1 + max(np.abs(coeffs[1:])) / np.abs(coeffs[0])
        roots = np.linspace(R, -R, n, dtype=np.float64)
        
        # Use JIT-compiled Aberth solver
        roots = self._aberth_solver(coeffs, roots)
        return np.sort(roots)[::-1].tolist()
    
    def _aberth_solver(self, coeffs, roots):
        return _aberth_impl(coeffs, roots)

@numba.njit
def _aberth_impl(coeffs, roots):
    n = len(roots)
    tolerance = 1e-6
    max_iter = 100
    
    for _ in range(max_iter):
        new_roots = np.zeros_like(roots)
        max_correction = 0.0
        converged = np.zeros(n, dtype=np.bool_)
        
        # Evaluate polynomial and derivative for all roots
        p_vals = np.zeros(n)
        dp_vals = np.zeros(n)
        for i in range(n):
            p, dp = horner(coeffs, roots[i])
            p_vals[i] = p
            dp_vals[i] = dp
            if np.abs(p) < 1e-12:
                converged[i] = True
                new_roots[i] = roots[i]
        
        # Compute reciprocal differences using broadcasting
        for i in range(n):
            if converged[i]:
                continue
                
            s = 0.0
            for j in range(n):
                if i != j:
                    diff = roots[i] - roots[j]
                    if np.abs(diff) > 1e-12:
                        s += 1.0 / diff
            
            denom = dp_vals[i] / p_vals[i] - s
            if np.abs(denom) < 1e-12:
                correction = 0.0
            else:
                correction = 1.0 / denom
                
            new_roots[i] = roots[i] - correction
            if np.abs(correction) > max_correction:
                max_correction = np.abs(correction)
        
        if max_correction < tolerance:
            return new_roots
        roots = new_roots
        
    return roots

@numba.njit
def horner(coeffs, x):
    n = len(coeffs) - 1
    p = coeffs[0]
    dp = 0.0
    for i in range(1, n+1):
        dp = dp * x + p
        p = p * x + coeffs[i]
    return p, dp