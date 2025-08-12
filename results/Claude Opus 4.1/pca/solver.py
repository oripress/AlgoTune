import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """Optimized PCA with algorithm selection based on problem size."""
        X = np.asarray(problem["X"], dtype=np.float32)  # Use float32 for speed
        n_components = problem["n_components"]
        m, n = X.shape
        
        # Center the data in-place
        mean = X.mean(axis=0)
        X -= mean
        
        # Choose algorithm based on dimensions
        if n_components < min(m, n) * 0.5 and n_components < 50 and m * n > 1000:
            # Use sparse SVD for small n_components relative to data size
            # This is much faster for large matrices with small n_components
            from scipy.sparse.linalg import svds
            _, _, V = svds(X, k=n_components)
            # svds returns V in reverse order
            V = V[::-1]
        else:
            # Use full SVD for small matrices or when n_components is large
            _, _, V = svd(X, full_matrices=False, overwrite_a=True, check_finite=False)
            V = V[:n_components]
        
        return V.astype(np.float64)  # Convert back to float64 for accuracy