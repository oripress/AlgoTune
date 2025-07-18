import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, ArpackError
import faiss
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        A faster implementation of Spectral Clustering using `eigsh` and FAISS KMeans.
        """
        similarity_matrix = np.array(problem["similarity_matrix"], dtype=np.float64)
        n_clusters = problem["n_clusters"]
        n_samples = similarity_matrix.shape[0]

        if n_samples == 0:
            return {"labels": [], "n_clusters": n_clusters}

        # Handle trivial cases
        if n_clusters <= 0:
            return {"labels": [0] * n_samples, "n_clusters": 1}
        if n_clusters == 1:
            return {"labels": [0] * n_samples, "n_clusters": 1}
        if n_clusters >= n_samples:
            labels = np.arange(n_samples)
            return {"labels": labels.tolist(), "n_clusters": n_clusters}

        # 1. Compute matrix for eigen-decomposition (related to Normalized Laplacian)
        # Using symmetric normalized Laplacian: L_sym = I - D^(-1/2) A D^(-1/2)
        # We find eigenvectors of D^(-1/2) A D^(-1/2)
        degree = np.sum(similarity_matrix, axis=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            d_inv_sqrt = 1.0 / np.sqrt(degree)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0
        
        # This is an efficient way to compute M = D^(-1/2) * A * D^(-1/2)
        M = (d_inv_sqrt[:, np.newaxis] * similarity_matrix) * d_inv_sqrt[np.newaxis, :]

        # 2. Eigen-decomposition of M
        try:
            # Use eigsh for efficiency. We need the k largest eigenvectors of M.
            eigenvalues, eigenvectors = eigsh(M, k=n_clusters, which='LM', tol=1e-4, v0=np.ones(n_samples), maxiter=2000)
        except (ArpackError, ValueError):
            # Fallback to eigh if eigsh fails (slower but more robust)
            eigenvalues, eigenvectors = eigh(M)
            # Get the top n_clusters eigenvectors
            indices = np.argsort(eigenvalues)[-n_clusters:]
            eigenvectors = eigenvectors[:, indices]

        # 3. Normalize the embedding vectors (rows) to unit length
        norm = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
        # Avoid division by zero
        norm[norm < 1e-9] = 1.0
        embedding_normalized = eigenvectors / norm

        # 4. K-Means clustering using FAISS for speed
        embedding_normalized = embedding_normalized.astype('float32')
        d = embedding_normalized.shape[1]
        
        if d == 0:
            # This can happen if n_clusters=0, though we handled that.
            # Or if eigenvectors are all zero for some reason.
            return {"labels": [0] * n_samples, "n_clusters": 1}

        kmeans = faiss.Kmeans(d, n_clusters, niter=20, nredo=3, seed=42, verbose=False)
        kmeans.train(embedding_normalized)
        _, labels = kmeans.index.search(embedding_normalized, 1)
        
        solution = {"labels": labels.flatten().tolist(), "n_clusters": n_clusters}
        return solution