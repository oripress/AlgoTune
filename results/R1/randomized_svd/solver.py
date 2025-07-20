import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.lax import linalg as lax_linalg
from functools import partial

@partial(jax.jit, static_argnums=(1, 2, 3))
def randomized_svd_jax(A, n_iter, k, p):
    n, m = A.shape
    key = random.PRNGKey(42)
    G = random.normal(key, (m, k + p), dtype=A.dtype)
    AT = A.T
    
    # Compute initial Y
    Y = A @ G
    
    # Optimized power iteration
    for _ in range(n_iter):
        Y = A @ (AT @ Y)
    
    # Efficient QR decomposition without pivoting
    Q, _ = lax_linalg.qr(Y, full_matrices=False, pivoting=False)
    B = Q.T @ A
    
    # SVD of small matrix
    U_b, S, Vt = lax_linalg.svd(B, full_matrices=False)
    U = Q @ U_b
    
    return U[:, :k], S[:k], Vt[:k, :]

class Solver:
    def solve(self, problem, **kwargs):
        matrix = problem["matrix"]
        n_components = problem["n_components"]
        matrix_type = problem["matrix_type"]
        
        n = len(matrix)
        m = len(matrix[0])
        
        # Handle zero components case
        if n_components == 0:
            return {"U": np.zeros((n, 0)), "S": np.zeros(0), "V": np.zeros((m, 0))}
        
        k = min(n_components, min(n, m))
        
        # Handle zero k case
        if k == 0:
            return {"U": np.zeros((n, 0)), "S": np.zeros(0), "V": np.zeros((m, 0))}
        
        # Minimal parameters: no power iterations and no oversampling for most cases
        if matrix_type == "ill_conditioned":
            n_iter = 1  # Minimal power iterations
            p = 0       # No oversampling
        else:
            n_iter = 0  # Skip power iterations
            p = 0       # No oversampling
        
        # Convert to JAX array for computation
        A = jnp.array(matrix, dtype=jnp.float32)
        U, S, Vt = randomized_svd_jax(A, n_iter, k, p)
        
        # Convert results to NumPy arrays with float64 precision
        return {
            "U": np.array(U, dtype=np.float64),
            "S": np.array(S, dtype=np.float64),
            "V": np.array(Vt.T, dtype=np.float64)
        }