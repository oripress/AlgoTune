import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from typing import Any

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        A = np.array(problem["covariance"], dtype=float)
        n_components = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])

        n = A.shape[0]
        k = min(n, n_components)
        
        if n < 128 or k == n:
            eigvals, eigvecs = np.linalg.eigh(A)
            top_eigvals = eigvals[-k:][::-1]
            top_eigvecs = eigvecs[:, -k:][:, ::-1]
        elif k <= n // 4:
            eigvals, eigvecs = scipy.sparse.linalg.eigsh(A, k=k, which='LA')
            top_eigvals = eigvals[::-1]
            top_eigvecs = eigvecs[:, ::-1]
        else:
            eigvals, eigvecs = scipy.linalg.eigh(A, subset_by_index=[n - k, n - 1])
            top_eigvals = eigvals[::-1]
            top_eigvecs = eigvecs[:, ::-1]
        
        # Keep only positive eigenvalues
        pos_mask = top_eigvals > 0
        top_eigvals = top_eigvals[pos_mask]
        top_eigvecs = top_eigvecs[:, pos_mask]
        
        # Scale eigenvectors in place
        np.sqrt(top_eigvals, out=top_eigvals)
        top_eigvecs *= top_eigvals
        
        # Soft-thresholding
        threshold = sparsity_param * 0.5
        X = np.copysign(np.maximum(np.abs(top_eigvecs) - threshold, 0.0), top_eigvecs)
        
        # Project onto unit ball in place
        norms = np.linalg.norm(X, axis=0)
        X /= np.maximum(norms, 1.0)
        
        # Pad with zeros if necessary
        k_pos = len(top_eigvals)
        if k_pos < n_components:
            X_full = np.zeros((n, n_components))
            X_full[:, :k_pos] = X
            X = X_full
            
        # Calculate explained variance
        AX = A @ X
        explained_variance = (X * AX).sum(axis=0).tolist()
        
        return {
            "components": X.tolist(),
            "explained_variance": explained_variance
        }