import numpy as np
from scipy.linalg import cholesky, solve_triangular

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the generalized eigenvalue problem A·x = λ·B·x.
        Transform to standard eigenvalue problem using Cholesky decomposition.
        """
        A, B = problem
        
        # Cholesky decomposition: B = L @ L^T
        L = cholesky(B, lower=True, check_finite=False, overwrite_a=False)
        
        # Solve L @ Y = A using triangular solve to get Y = L^{-1} @ A
        Y = solve_triangular(L, A, lower=True, check_finite=False)
        
        # Compute Atilde = L^{-1} @ A @ L^{-T} = Y @ L^{-T}
        # Solve L @ X = Y^T to get X = L^{-1} @ Y^T
        # Then Atilde = X^T = (L^{-1} @ Y^T)^T = Y @ L^{-T}
        X = solve_triangular(L, Y.T, lower=True, check_finite=False)
        Atilde = X.T
        
        # Use numpy's eigvalsh which can be faster
        eigenvalues = np.linalg.eigvalsh(Atilde)
        
        # eigvalsh returns in ascending order, reverse for descending
        return eigenvalues[::-1].tolist()