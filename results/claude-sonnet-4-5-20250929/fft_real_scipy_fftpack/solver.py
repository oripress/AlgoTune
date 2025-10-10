import jax.numpy as jnp
import numpy as np
from jax import jit

class Solver:
    def __init__(self):
        """Initialize with JIT-compiled FFT function."""
        self._fft_func = jit(jnp.fft.fftn)

    def solve(self, problem, **kwargs):
        """
        Use JAX's optimized FFT with JIT compilation.
        """
        # Convert to JAX array, compute FFT, convert back to numpy
        result = self._fft_func(problem)
        return np.asarray(result)