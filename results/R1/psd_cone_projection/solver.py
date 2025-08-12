import jax.numpy as jnp
from jax import jit
import numpy as np
from typing import Any

@jit
def project_psd(A):
    """JAX-accelerated PSD projection."""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.maximum(eigvals, 0.0)
    X = (eigvecs * eigvals) @ eigvecs.T
    # Ensure symmetry to handle numerical precision issues
    return (X + X.T) / 2

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """Optimized PSD projection using JAX."""
        # Convert input to JAX array with float64 precision
        A = jnp.array(problem["A"], dtype=jnp.float64)
        
        # Compute projection
        X = project_psd(A)
        
        # Convert to NumPy array and enforce symmetry again
        X_np = np.array(X, dtype=np.float64)
        X_np = (X_np + X_np.T) / 2
        
        # Return as list of lists
        return {"X": X_np.tolist()}