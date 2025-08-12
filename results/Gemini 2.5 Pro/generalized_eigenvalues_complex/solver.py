# Attempt to import JAX and configure it for high-performance computation.
try:
    import jax
    from jax.config import config
    # Enable 64-bit precision for accuracy, matching SciPy's default.
    config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    JAX_AVAILABLE = True

    # Define the JIT-compiled solver function at the module level.
    # This allows JAX to compile the function once and reuse the optimized
    # XLA code for subsequent calls.
    @jax.jit
    def _solve_jax(A, B):
        """
        JIT-compiled function to solve the generalized eigenvalue problem
        by converting it to a standard one: (B⁻¹A)x = λx.
        """
        # 1. Compute C = B⁻¹A. Using linalg.solve(B, A) is more numerically
        #    stable than computing the inverse of B directly.
        C = jnp.linalg.solve(B, A)
        
        # 2. Solve the standard eigenvalue problem for C.
        #    jax.numpy.linalg.eig returns a tuple of (eigenvalues, eigenvectors).
        eigenvalues = jnp.linalg.eig(C)[0]
        
        # 3. Sort in descending order using the negation trick.
        return -jnp.sort(-eigenvalues)

except ImportError:
    JAX_AVAILABLE = False

# Import CPU libraries for the primary or fallback path.
import numpy as np
import scipy.linalg as la
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solves the generalized eigenvalue problem using a JAX/SciPy hybrid strategy.
        """
        A_np, B_np = problem
        n = A_np.shape[0]

        # Heuristic: Use JAX/GPU only for matrix sizes where it's likely to be
        # fastest, avoiding JIT/transfer overhead for small N and potential
        # performance cliffs for very large N.
        use_jax = JAX_AVAILABLE and 256 <= n <= 2500

        if use_jax:
            try:
                # JAX Path:
                # JAX implicitly handles data movement to the accelerator (GPU/TPU).
                result_jax = _solve_jax(A_np, B_np)
                # Block until computation is finished and convert to a NumPy array.
                result_np = np.array(result_jax)
                return result_np.tolist()
            except Exception:
                # Fallback to CPU if the JAX path fails for any reason.
                pass

        # CPU Path (primary for small/large N, or fallback):
        eigenvalues = la.eig(
            A_np, B_np,
            right=False,
            overwrite_a=True,
            overwrite_b=True,
            check_finite=False
        )
        
        solution_np = -np.sort(-eigenvalues)
        return solution_np.tolist()