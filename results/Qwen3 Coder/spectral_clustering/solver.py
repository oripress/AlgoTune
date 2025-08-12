import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import eigh
import numba
from numba import jit

@jit(nopython=True, fastmath=True)
def compute_normalized_matrix(similarity_matrix, degree_sqrt_inv):
    """Compute normalized matrix: D^(-1/2) * W * D^(-1/2) using numba"""
    n = similarity_matrix.shape[0]
    normalized_matrix = np.zeros((n, n), dtype=similarity_matrix.dtype)
    
    for i in range(n):
        for j in range(n):
            normalized_matrix[i, j] = similarity_matrix[i, j] * degree_sqrt_inv[i] * degree_sqrt_inv[j]
    
    return normalized_matrix

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the spectral clustering problem using a custom implementation.
        
        :param problem: A dictionary representing the spectral clustering problem.
                       Requires keys: "similarity_matrix", "n_clusters".
        :return: A dictionary containing the solution:
                "labels": numpy array of predicted cluster labels.
        """
        similarity_matrix = np.asarray(problem["similarity_matrix"], dtype=np.float64)
        n_clusters = problem["n_clusters"]
        n_samples = similarity_matrix.shape[0]
        
        # Handle edge cases
        if n_clusters >= n_samples:
            labels = np.arange(n_samples)
        elif n_samples == 0:
            labels = np.array([], dtype=int)
        elif n_clusters <= 1:
            labels = np.zeros(n_samples, dtype=int)
        else:
            # Compute degree vector (sum of each row)
            degree = np.sum(similarity_matrix, axis=1)
            # Avoid division by zero
            degree = np.maximum(degree, 1e-10)
            
            # Compute D^(-1/2) as a vector (more efficient than diagonal matrix)
            degree_sqrt_inv = np.power(degree, -0.5)
            
            # Compute normalized matrix: D^(-1/2) * W * D^(-1/2)
            if n_samples < 200:  # Use numba for smaller matrices
                normalized_matrix = compute_normalized_matrix(similarity_matrix, degree_sqrt_inv)
            else:
                # For larger matrices, use numpy broadcasting
                normalized_matrix = similarity_matrix * np.outer(degree_sqrt_inv, degree_sqrt_inv)
            
            # The normalized Laplacian is I - normalized_matrix
            identity = np.eye(n_samples, dtype=np.float64)
            normalized_laplacian = identity - normalized_matrix
            
            # Compute eigenvalues and eigenvectors
            # For efficiency, compute only what we need
            try:
                # Use subset_by_index for efficiency if available
                eigenvalues, eigenvectors = eigh(normalized_laplacian, subset_by_index=[0, n_clusters-1])
            except TypeError:
                # Fallback for older scipy versions
                eigenvalues, eigenvectors = eigh(normalized_laplacian)
                eigenvalues = eigenvalues[:n_clusters]
                eigenvectors = eigenvectors[:, :n_clusters]
            
            # Apply k-means clustering on the eigenvectors
            kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
            labels = kmeans.fit_predict(eigenvectors)
        
        return {"labels": labels.tolist()}