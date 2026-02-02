import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from numpy.typing import NDArray
from typing import Any

class Solver:
    def __init__(self):
        # JIT compile the FFT function at init time
        self._fftn = jit(jnp.fft.fftn)
        # Single warmup to ensure compiled
        _ = self._fftn(jnp.zeros((4, 4))).block_until_ready()
    
    def solve(self, problem: NDArray, **kwargs) -> Any:
        """
        Compute the N-dimensional FFT of a real-valued matrix.
        """
        # Convert to JAX array first
        arr = jnp.asarray(problem)
        result = self._fftn(arr)
        return np.asarray(result)