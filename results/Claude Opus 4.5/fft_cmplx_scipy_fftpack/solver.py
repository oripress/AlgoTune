import numpy as np
import jax.numpy as jnp
import jax
import scipy.fft
from numpy.typing import NDArray

class Solver:
    def __init__(self):
        # JIT compile the FFT function
        self._fft_fn = jax.jit(jnp.fft.fftn)
        # Warm up the JIT with a small array
        dummy = jnp.zeros((4, 4), dtype=jnp.complex128)
        self._fft_fn(dummy).block_until_ready()
    
    def solve(self, problem: NDArray, **kwargs) -> NDArray:
        """
        Compute the N-dimensional FFT using JAX for large arrays, scipy for small.
        """
        # For very small arrays, scipy has less overhead
        if problem.size < 256:
            return scipy.fft.fftn(problem)
        
        # Use device_put for more efficient transfer
        arr = jax.device_put(problem)
        result = self._fft_fn(arr)
        # Use jax.device_get for efficient transfer back
        return jax.device_get(result)