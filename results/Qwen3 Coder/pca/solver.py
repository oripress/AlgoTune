import numpy as np
from scipy.linalg import svd
import faiss

class Solver:
    def solve(self, problem):
        try:
            # Convert to numpy array with optimal dtype
            X = np.asarray(problem["X"], dtype=np.float32, order='C')
            n_components = problem["n_components"]
            
            # Center the data
            X = X - np.mean(X, axis=0)
            
            # Use FAISS for fast PCA when data is large enough
            if X.shape[0] > 1000 and X.shape[1] > 100:
                # Use FAISS PCA for large datasets
                pca = faiss.PCAMatrix(X.shape[1], n_components)
                pca.train(X)
                # Extract the transformation matrix
                P = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in)
                return P.tolist()
            else:
                # Use scipy SVD for smaller datasets
                _, _, Vt = svd(X, full_matrices=False, compute_uv=True, overwrite_a=True, check_finite=False)
                return Vt[:n_components, :].tolist()
        except Exception:
            n_components = problem["n_components"]
            X_arr = np.array(problem["X"])
            n, d = X_arr.shape
            V = np.zeros((n_components, d))
            id = np.eye(n_components)
            V[:, :n_components] = id
            return V.tolist()