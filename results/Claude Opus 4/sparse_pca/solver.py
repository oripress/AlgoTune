import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from scipy.linalg import eigh

@jit
def soft_threshold_jax(x, threshold):
    """Vectorized soft thresholding operator."""
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0)

@jit
def project_columns_unit_ball(X):
    """Project each column of X onto unit ball."""
    norms = jnp.linalg.norm(X, axis=0, keepdims=True)
    return X / jnp.maximum(norms, 1.0)

@jit
def fista_step(X, Y, X_old, t, B, step_size, sparsity_param):
    """Single FISTA iteration."""
    # Gradient at Y
    grad = 2 * (Y - B)
    
    # Gradient step
    X_temp = Y - step_size * grad
    
    # Proximal step (soft thresholding)
    X_new = soft_threshold_jax(X_temp, step_size * sparsity_param)
    
    # Project onto unit ball constraints
    X_new = project_columns_unit_ball(X_new)
    
    # FISTA momentum update
    t_new = (1 + jnp.sqrt(1 + 4 * t * t)) / 2
    Y_new = X_new + ((t - 1) / t_new) * (X_new - X)
    
    return X_new, Y_new, t_new

@jit
def compute_explained_variance(X, A):
    """Compute explained variance for each component."""
    # Vectorized computation: diag(X.T @ A @ X)
    return jnp.sum(X * (A @ X), axis=0)

class Solver:
    def __init__(self):
        # Pre-compile the JIT functions
        self.fista_step_compiled = fista_step
        self.compute_variance_compiled = compute_explained_variance
        
    def solve(self, problem: dict) -> dict:
        """
        Solve the sparse PCA problem using JAX-accelerated FISTA.
        
        :param problem: Dictionary with problem parameters
        :return: Dictionary with the sparse principal components
        """
        A = np.array(problem["covariance"], dtype=np.float32)
        n_components = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])
        
        n = A.shape[0]
        
        # Get eigendecomposition using scipy (faster than JAX for this)
        eigvals, eigvecs = eigh(A)
        
        # Keep only positive eigenvalues
        pos_mask = eigvals > 0
        if np.any(pos_mask):
            eigvals = eigvals[pos_mask]
            eigvecs = eigvecs[:, pos_mask]
            
            # Sort in descending order
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            
            # Use top n_components eigenvectors scaled by sqrt(eigenvalues)
            k = min(len(eigvals), n_components)
            B = eigvecs[:, :k] * np.sqrt(eigvals[:k])
        else:
            B = np.zeros((n, n_components))
        
        # Pad B if necessary
        if B.shape[1] < n_components:
            B = np.pad(B, ((0, 0), (0, n_components - B.shape[1])), mode='constant')
        
        # Convert to JAX arrays
        B = jnp.array(B, dtype=jnp.float32)
        A_jax = jnp.array(A, dtype=jnp.float32)
        
        # Initialize
        X = B
        Y = X
        t = 1.0
        
        # FISTA parameters
        L = 2.0  # Lipschitz constant
        step_size = 1.0 / L
        max_iter = 300  # Reduced iterations since JAX is faster
        tol = 1e-5
        
        # Run FISTA
        for i in range(max_iter):
            X_old = X
            X, Y, t = self.fista_step_compiled(X, Y, X_old, t, B, step_size, sparsity_param)
            
            # Check convergence every 10 iterations to reduce overhead
            if i % 10 == 0:
                diff_norm = jnp.linalg.norm(X - X_old)
                if diff_norm < tol:
                    break
        
        # Calculate explained variance
        explained_variance = self.compute_variance_compiled(X, A_jax)
        
        # Convert back to Python lists
        return {
            "components": np.array(X).tolist(),
            "explained_variance": np.array(explained_variance).tolist()
        }