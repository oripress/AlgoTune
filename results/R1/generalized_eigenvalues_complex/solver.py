import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu_solve, lu_factor

class Solver:
    def solve(self, problem, **kwargs):
        """Solve the generalized eigenvalue problem A·x = λB·x using JAX acceleration."""
        A, B = problem
        
        # Convert to JAX arrays
        A = jnp.array(A, dtype=jnp.float64)
        B = jnp.array(B, dtype=jnp.float64)
        
        # Compute LU decomposition of B for efficient solving
        lu, piv = lu_factor(B)
        
        # Solve the generalized eigenvalue problem
        eigenvalues = jnp.linalg.eigvals(lu_solve((lu, piv), A))
        
        # Vectorized sorting: descending order by real part, then by imaginary part
        sorted_indices = jnp.lexsort(jnp.array([-eigenvalues.imag, -eigenvalues.real]))
        return eigenvalues[sorted_indices].tolist()