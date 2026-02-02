import numpy as np
from scipy.special import wright_bessel
from numpy.polynomial.legendre import leggauss

class Solver:
    def __init__(self):
        # Pre-compute Gauss-Legendre quadrature nodes and weights
        # Higher order for better accuracy  
        self.nodes, self.weights = leggauss(250)
    
    def solve(self, problem, **kwargs):
        a = np.array(problem["a"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        lower = np.array(problem["lower"], dtype=np.float64)
        upper = np.array(problem["upper"], dtype=np.float64)
        
        n = len(a)
        if n == 0:
            return {"result": []}
        
        results = np.zeros(n)
        nodes = self.nodes
        weights = self.weights
        
        for i in range(n):
            # Transform from [-1, 1] to [lower, upper]
            mid = (upper[i] + lower[i]) / 2.0
            half_len = (upper[i] - lower[i]) / 2.0
            x_vals = mid + half_len * nodes
            
            # tanhsinh calls wright_bessel(x, a, b) with args=(a, b)
            f_vals = wright_bessel(x_vals, a[i], b[i])
            
            # Compute integral
            results[i] = half_len * np.dot(weights, f_vals)
        
        return {"result": results.tolist()}