import jax
import jax.numpy as jnp
from functools import lru_cache

class Solver:
    def __init__(self):
        """Pre-compile JIT functions for various polynomial degrees."""
        # Pre-compile for common sizes to avoid compilation overhead
        self.compute_roots_jit = jax.jit(self._compute_roots_core)
        # Warm up the JIT compiler with a small example
        dummy = jnp.array([1.0, 0.0], dtype=jnp.float64)
        _ = self.compute_roots_jit(dummy)
    
    @staticmethod
    def _compute_roots_core(coefficients):
        """Core root-finding function."""
        n = coefficients.shape[0] - 1
        
        # Build companion matrix
        companion = jnp.zeros((n, n), dtype=jnp.float64)
        companion = companion.at[0, :].set(-coefficients[1:] / coefficients[0])
        companion = companion.at[jnp.arange(1, n), jnp.arange(0, n-1)].set(1.0)
        
        # Compute eigenvalues
        computed_roots = jnp.linalg.eigvals(companion)
        computed_roots = jnp.real(computed_roots)
        computed_roots = jnp.sort(computed_roots)[::-1]
        
        return computed_roots
    
    def solve(self, problem, **kwargs):
        """
        Optimized polynomial root finder using JAX JIT compilation with pre-warming.
        """
        n = len(problem) - 1
        
        if n == 0:
            return []
        
        coefficients = jnp.array(problem, dtype=jnp.float64)
        computed_roots = self.compute_roots_jit(coefficients)
        
        return computed_roots.tolist()