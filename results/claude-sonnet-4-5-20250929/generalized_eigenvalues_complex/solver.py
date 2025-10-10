import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
from jax import jit

@jit
def compute_eigenvalues(A, B):
    """JIT compiled eigenvalue computation."""
    invB_A = jnp.linalg.solve(B, A)
    eigenvalues = jnp.linalg.eigvals(invB_A)
    
    # Sort eigenvalues using jax operations
    real_parts = -eigenvalues.real
    imag_parts = -eigenvalues.imag
    indices = jnp.lexsort((imag_parts, real_parts))
    
    return eigenvalues[indices]

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[complex]:
        """
        Solve the generalized eigenvalue problem for the given matrices A and B.
        
        The problem is defined as: A · x = λ B · x.
        We convert this to inv(B) * A * x = λ * x
        
        :param problem: Tuple (A, B) where A and B are n x n real matrices.
        :return: List of eigenvalues (complex numbers) sorted in descending order.
        """
        A, B = problem
        
        # Convert to JAX arrays and compute
        eigenvalues = compute_eigenvalues(jnp.array(A), jnp.array(B))
        
        return eigenvalues.tolist()