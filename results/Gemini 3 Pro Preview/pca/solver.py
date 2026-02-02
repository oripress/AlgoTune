import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem, **kwargs):
        # Use float32 for speed
        X = np.array(problem["X"], dtype=np.float32)
        n_components = problem["n_components"]
        m, n = X.shape
        
        if n_components == 0:
            return np.zeros((0, n), dtype=np.float32)

        # Calculate mean
        mean = np.mean(X, axis=0)
        
        if m >= n:
            # Implicit centering for covariance matrix
            # C = (X - mean).T @ (X - mean) = X.T @ X - m * mean.T @ mean
            # Compute X.T @ X
            # We can use a temporary variable for X.T @ X to save memory if needed, 
            # but here we just compute it.
            
            # Note: Precision issues with float32 and implicit centering might occur 
            # if data is far from 0. However, let's try.
            # If precision is bad, we might need to center explicitly.
            # Let's center explicitly to be safe, but optimize it.
            
            X -= mean
            
            # Compute Covariance
            # overwrite_a=True allows eigh to destroy C, saving memory/time
            C = np.dot(X.T, X)
            
            subset = (n - n_components, n - 1)
            # check_finite=False for speed
            evals, evecs = scipy.linalg.eigh(C, subset_by_index=subset, 
                                             overwrite_a=True, check_finite=False)
            
            # evecs are columns, corresponding to eigenvalues in ascending order.
            # We need descending order for Vt rows.
            Vt = evecs.T[::-1]
            
        else:
            # n > m case (High dimension, few samples)
            # Center explicitly
            X -= mean
            
            # Gram matrix K = X @ X.T
            K = np.dot(X, X.T)
            
            subset = (m - n_components, m - 1)
            evals, U = scipy.linalg.eigh(K, subset_by_index=subset, 
                                         overwrite_a=True, check_finite=False)
            
            # Sort descending
            U = U[:, ::-1]
            evals = evals[::-1]
            
            # Vt = S^-1 U^T X
            # S = sqrt(evals)
            # Avoid division by zero
            S_inv = 1.0 / np.sqrt(np.maximum(evals, 1e-10))
            
            # Compute Vt
            # U.T is (n_components, m)
            # X is (m, n)
            # Vt is (n_components, n)
            Vt = np.dot(U.T, X)
            Vt *= S_inv[:, np.newaxis]
            
        return Vt