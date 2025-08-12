from typing import Any
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize
import numba
import warnings
warnings.filterwarnings('ignore')
@numba.jit(nopython=True, fastmath=True)
def compute_normalized_laplacian_vectorized(similarity_matrix, degree_inv_sqrt, n_samples):
    """Compute normalized Laplacian using vectorized operations in Numba with fastmath."""
    laplacian_norm = np.zeros((n_samples, n_samples), dtype=np.float32)
    
    # Vectorized computation: D^(-1/2) * W * D^(-1/2)
    for i in range(n_samples):
        d_i = degree_inv_sqrt[i]
        if d_i > 0:
            # Compute row i: d_i * W[i, :] * degree_inv_sqrt
            for j in range(n_samples):
                d_j = degree_inv_sqrt[j]
                if d_j > 0:
                    laplacian_norm[i, j] = -d_i * similarity_matrix[i, j] * d_j
            laplacian_norm[i, i] += 1.0
    
    return laplacian_norm

def compute_eigenvectors_very_fast(laplacian_norm, n_clusters, n_samples):
    """Extremely optimized eigenvector computation."""
    if n_samples <= 30:
        # Very small matrices - use numpy directly
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_norm)
        eigenvectors = eigenvectors[:, 1:n_clusters+1] if n_clusters < n_samples else eigenvectors[:, :n_clusters]
    elif n_samples <= 200:
        # Medium matrices - use scipy's eigh with subset for speed
        try:
            eigenvalues, eigenvectors = eigh(laplacian_norm, subset_by_index=[0, min(n_clusters, n_samples-1)])
            # Skip the first eigenvector (constant vector) if needed
            if eigenvalues[0] < 1e-10 and n_clusters < n_samples:
                eigenvectors = eigenvectors[:, 1:]
        except:
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian_norm)
            eigenvectors = eigenvectors[:, 1:n_clusters+1] if n_clusters < n_samples else eigenvectors[:, :n_clusters]
    else:
        # Larger matrices - use power method for extreme speed
        k = min(n_clusters + 1, n_samples - 1)
        
        # Use power method for the first few eigenvectors
        try:
            eigenvectors = np.zeros((n_samples, k), dtype=np.float32)
            
            # Power method for each eigenvector
            for i in range(k):
                # Random initial vector
                v = np.random.randn(n_samples).astype(np.float32)
                v = v / np.linalg.norm(v)
                
                # Power iteration with very few steps
                for _ in range(10):  # Very few iterations
                    v_new = laplacian_norm @ v
                    # Orthogonalize against previous eigenvectors
                    for j in range(i):
                        v_new -= np.dot(v_new, eigenvectors[:, j]) * eigenvectors[:, j]
                    v_norm = np.linalg.norm(v_new)
                    if v_norm > 1e-10:
                        v = v_new / v_norm
                    else:
                        break
                
                eigenvectors[:, i] = v
            
            # Skip the first eigenvector if it's constant
            if k > 1 and np.allclose(eigenvectors[:, 0], np.ones(n_samples) / np.sqrt(n_samples)):
                eigenvectors = eigenvectors[:, 1:]
                
        except:
            # Fallback to sparse computation
            laplacian_sparse = csr_matrix(laplacian_norm)
            eigenvalues, eigenvectors = eigsh(
                laplacian_sparse, 
                k=k, 
                which='SM', 
                maxiter=50,
                tol=1e-1,
                v0=None
            )
            if eigenvalues[0] < 1e-10 and n_clusters < n_samples:
                eigenvectors = eigenvectors[:, 1:]
                eigenvectors = eigenvectors[:, 1:]
    
    return eigenvectors

@numba.jit(nopython=True, fastmath=True)
def normalize_eigenvectors_numba(eigenvectors):
    """Normalize eigenvectors by row norm using Numba."""
    n_samples, n_features = eigenvectors.shape
    eigenvectors_norm = np.empty_like(eigenvectors)
    
    for i in range(n_samples):
        row_norm = 0.0
        for j in range(n_features):
            row_norm += eigenvectors[i, j] ** 2
        
        if row_norm > 0:
            row_norm = np.sqrt(row_norm)
            for j in range(n_features):
                eigenvectors_norm[i, j] = eigenvectors[i, j] / row_norm
        else:
            for j in range(n_features):
                eigenvectors_norm[i, j] = 0.0
    
    return eigenvectors_norm

@numba.jit(nopython=True, fastmath=True)
def kmeans_simple_numba(data, n_clusters, max_iter=5):
    """Very simple k-means implementation in Numba for speed."""
    n_samples, n_features = data.shape
    
    # Initialize centroids randomly
    np.random.seed(42)
    centroids = np.empty((n_clusters, n_features), dtype=data.dtype)
    for k in range(n_clusters):
        idx = np.random.randint(0, n_samples)
        centroids[k] = data[idx]
    
    # Very few iterations for speed
    for _ in range(max_iter):
        # Assign points to nearest centroid
        labels = np.empty(n_samples, dtype=np.int32)
        for i in range(n_samples):
            min_dist = np.inf
            best_k = 0
            for k in range(n_clusters):
                dist = 0.0
                for j in range(n_features):
                    diff = data[i, j] - centroids[k, j]
                    dist += diff * diff
                if dist < min_dist:
                    min_dist = dist
                    best_k = k
            labels[i] = best_k
        
        # Update centroids
        for k in range(n_clusters):
            count = 0
            for j in range(n_features):
                centroids[k, j] = 0.0
            
            for i in range(n_samples):
                if labels[i] == k:
                    count += 1
                    for j in range(n_features):
                        centroids[k, j] += data[i, j]
            
            if count > 0:
                for j in range(n_features):
                    centroids[k, j] /= count
    
    return labels
def orthogonal_iteration(matrix, k, n_samples, max_iter=20):
    """Orthogonal iteration to find k smallest eigenvectors."""
    # Initialize random orthogonal matrix
    np.random.seed(42)
    Q = np.random.randn(n_samples, k).astype(np.float32)
    Q, _ = np.linalg.qr(Q)
    
    # Shift matrix to find smallest eigenvalues
    shift = 0.1
    shifted_matrix = matrix - shift * np.eye(n_samples, dtype=np.float32)
    
    for _ in range(max_iter):
        # Power iteration
        Q = shifted_matrix @ Q
        
        # Orthogonalize
        Q, _ = np.linalg.qr(Q)
    
    # Compute Rayleigh quotients
    eigenvalues = np.diag(Q.T @ matrix @ Q)
    
    # Sort by eigenvalues
    idx = np.argsort(eigenvalues)
    Q = Q[:, idx]
    
    return Q

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the spectral clustering problem using a custom implementation.

        :param problem: A dictionary representing the spectral clustering problem.
        :return: A dictionary containing the solution with "labels" key.
        """
        # Extract inputs directly to avoid dictionary lookups
        similarity_matrix = problem["similarity_matrix"]
        n_clusters = problem["n_clusters"]
        n_samples = len(similarity_matrix)
        
        # Handle edge cases efficiently
        if n_clusters >= n_samples:
            return {"labels": list(range(n_samples))}
        
        if n_samples == 0:
            return {"labels": []}
        
        # Convert to numpy array with optimal dtype
        similarity_matrix = np.array(similarity_matrix, dtype=np.float32)

        # Compute degree matrix
        degree = np.sum(similarity_matrix, axis=1)
        
        # Compute normalized Laplacian: L = I - D^(-1/2) * W * D^(-1/2)
        degree_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
        mask = degree > 0
        degree_inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
        
        # Use Numba for Laplacian computation
        laplacian_norm = compute_normalized_laplacian_vectorized(similarity_matrix, degree_inv_sqrt, n_samples)

        # Compute eigenvectors using very fast function
        eigenvectors = compute_eigenvectors_very_fast(laplacian_norm, n_clusters, n_samples)

        # Normalize eigenvectors by row norm
        if n_samples > 200:
            # Use Numba for large matrices
            eigenvectors_norm = normalize_eigenvectors_numba(eigenvectors.astype(np.float32))
        else:
            # Use numpy for smaller matrices
            row_norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
            row_norms[row_norms == 0] = 1
            eigenvectors_norm = eigenvectors / row_norms

        # Apply k-means clustering with aggressive optimization
        if n_samples > 1000:
            # For very large datasets, use MiniBatchKMeans
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, 
                init='k-means++', 
                n_init=1,
                batch_size=min(100, n_samples // 10),
                max_iter=10,  # Very reduced iterations
                random_state=42
            )
            labels = kmeans.fit_predict(eigenvectors_norm)
        elif n_samples > 200:
            # For large datasets, use optimized KMeans
            kmeans = KMeans(
                n_clusters=n_clusters, 
                init='k-means++', 
                n_init=1,  # Single initialization
                algorithm='lloyd', 
                max_iter=5,   # Very few iterations
                tol=1e-1,    # High tolerance
                random_state=42
            )
            labels = kmeans.fit_predict(eigenvectors_norm)
        else:
            # For smaller datasets, use Numba k-means
            labels = kmeans_simple_numba(eigenvectors_norm.astype(np.float32), n_clusters, max_iter=5)

        return {"labels": labels.tolist()}