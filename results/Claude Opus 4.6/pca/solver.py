import numpy as np
from scipy.linalg import svd, eigh, qr
from sklearn.utils.extmath import randomized_svd

class Solver:
    def solve(self, problem, **kwargs):
        X = np.array(problem["X"], dtype=np.float64)
        n_components = problem["n_components"]
        m, n = X.shape
        
        # Center the data
        X -= X.mean(axis=0)
        
        if n_components == 0:
            return np.zeros((0, n))
        
        k = min(m, n)
        
        if n_components >= k:
            # Full SVD needed
            U, s, Vt = svd(X, full_matrices=False)
            return Vt[:n_components]
        
        # Threshold for using randomized SVD vs exact methods
        use_randomized = (n_components <= min(m, n) // 5) and min(m, n) > 50
        
        if use_randomized:
            U, s, Vt = randomized_svd(X, n_components=n_components, n_iter=4, random_state=42)
            return Vt
        
        # Exact methods for when n_components is a significant fraction
        if m >= n:
            C = X.T @ X
            eigenvalues, eigenvectors = eigh(C, subset_by_index=[n - n_components, n - 1])
            V = eigenvectors[:, ::-1].T
            return V
        else:
            C = X @ X.T
            eigenvalues, eigenvectors = eigh(C, subset_by_index=[m - n_components, m - 1])
            eigenvectors = eigenvectors[:, ::-1]
            eigenvalues = eigenvalues[::-1]
            V = eigenvectors.T @ X
            norms = np.linalg.norm(V, axis=1, keepdims=True)
            norms[norms == 0] = 1
            V /= norms
            return V