import numpy as np
import jax
import jax.numpy as jnp

# Enable float64 in JAX
jax.config.update("jax_enable_x64", True)

class Solver:
    def __init__(self):
        # JIT compile the matmul operation
        self._matmul = jax.jit(jnp.matmul)
        # Warm up the JIT with various sizes
        for size in [10, 100, 500]:
            dummy_a = jnp.zeros((size, size), dtype=jnp.float64)
            dummy_b = jnp.zeros((size, size), dtype=jnp.float64)
            _ = self._matmul(dummy_a, dummy_b).block_until_ready()
    
    def solve(self, problem, **kwargs):
        A = problem["A"]
        B = problem["B"]
        # Convert to numpy first (faster than jnp.array from lists)
        if not isinstance(A, np.ndarray):
            A = np.asarray(A, dtype=np.float64)
        if not isinstance(B, np.ndarray):
            B = np.asarray(B, dtype=np.float64)
        # Use JAX for computation
        A_jax = jnp.asarray(A)
        B_jax = jnp.asarray(B)
        C = self._matmul(A_jax, B_jax)
        return np.asarray(C)