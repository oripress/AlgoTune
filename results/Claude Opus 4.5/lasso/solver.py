import numpy as np
from numpy.random import RandomState
from sklearn.linear_model._cd_fast import enet_coordinate_descent, enet_coordinate_descent_gram

class Solver:
    def __init__(self):
        self.rng = RandomState(0)
    
    def solve(self, problem, **kwargs):
        X = np.asarray(problem["X"], dtype=np.float64, order='F')
        y = np.ascontiguousarray(problem["y"], dtype=np.float64).ravel()
        
        n, d = X.shape
        
        if n == 0 or d == 0:
            return [0.0] * d if d > 0 else []
        
        # Regularization parameter scaled by n
        alpha_scaled = 0.1 * n
        l1_reg = alpha_scaled
        l2_reg = 0.0
        
        max_iter = 1000
        tol = 1e-4
        
        w = np.zeros(d, dtype=np.float64)
        
        # Use Gram matrix when computing X'X is cheaper than iterating through X
        # d*d + d*n (precompute) vs n*d*iterations (direct)
        # Typically precompute is better when d < n
        if d <= n and d < 1000:
            # Precompute Gram matrix and Xy using BLAS
            precompute = np.ascontiguousarray(np.dot(X.T, X))
            Xy = np.ascontiguousarray(np.dot(X.T, y))
            
            enet_coordinate_descent_gram(
                w, l1_reg, l2_reg, precompute, Xy, y, max_iter, tol, 
                self.rng, False, False
            )
        else:
            # Direct coordinate descent on X
            enet_coordinate_descent(
                w, l1_reg, l2_reg, X, y, max_iter, tol,
                self.rng, False, False
            )
        
        return w.tolist()