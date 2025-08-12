import numpy as np
from numpy.typing import NDArray
from typing import Any
import jax.numpy as jnp
from jax import jit

# JIT-compile the FFT function for better performance
@jit
def fftn_jit(x):
    return jnp.fft.fftn(x)

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> Any:
        """
        Compute the N-dimensional FFT using JAX with JIT compilation.
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        # Use JAX's FFT with JIT compilation for maximum performance
        if isinstance(problem, np.ndarray):
            # Convert to JAX array, compute FFT with JIT, and convert back
            jax_array = jnp.array(problem)
            result = fftn_jit(jax_array)
            return np.asarray(result)
        else:
            jax_array = jnp.array(problem)
            result = fftn_jit(jax_array)
            return np.asarray(result)