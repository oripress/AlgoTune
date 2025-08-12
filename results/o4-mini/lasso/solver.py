import numpy as np
import sklearn.linear_model._cd_fast as cd_fast
from numpy.random import RandomState

_ecdg = cd_fast.enet_coordinate_descent_gram
_ecd = cd_fast.enet_coordinate_descent
_dot = np.dot

class Solver:
    def __init__(self):
        # Initialize RNG once to avoid per-call overhead
        self.rng = RandomState(0)

    def solve(self, problem, **kwargs) -> list[float]:
        # Prepare data arrays; X in Fortran order for efficient column access
        X = np.asarray(problem["X"], dtype=np.float64, order='F')
        y = np.asarray(problem["y"], dtype=np.float64)
        n_samples, n_features = X.shape

        # Scale alpha for coordinate descent routines
        alpha = kwargs.get("alpha", 0.1)
        alpha_cd = alpha * n_samples
        beta_cd = 0.0  # no L2 penalty for Lasso

        # Iteration controls
        max_iter = kwargs.get("max_iter", 1000)
        tol = kwargs.get("tol", 1e-4)

        # Initialize coefficients in Fortran order
        w = np.zeros(n_features, dtype=np.float64, order='F')

        # Choose Gram-based or data-based coordinate descent
        if n_samples > n_features:
            # Precompute Gram matrix and X'y
            Gram = np.empty((n_features, n_features), dtype=np.float64, order='F')
            _dot(X.T, X, out=Gram)
            Xy = np.dot(X.T, y)
            # Fast C-level Gram-based coordinate descent
            _ecdg(w, alpha_cd, beta_cd, Gram, Xy, y, max_iter, tol, self.rng)
        else:
            # Fast C-level data-based coordinate descent
            _ecd(w, alpha_cd, beta_cd, X, y, max_iter, tol, self.rng)

        return w.tolist()