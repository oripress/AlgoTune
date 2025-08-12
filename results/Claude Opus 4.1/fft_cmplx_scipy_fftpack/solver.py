import jax.numpy as jnp
import jax
from typing import Any

class Solver:
    def __init__(self):
        # Configure JAX for best performance
        jax.config.update("jax_enable_x64", False)  # Use float32 for speed
        
        # Pre-compile the FFT function with JIT
        self.fft_func = jax.jit(jnp.fft.fftn)
    
    def solve(self, problem: Any, **kwargs) -> Any:
        """
        Compute the N-dimensional FFT using JAX with JIT compilation.
        """
        # Direct call to JIT-compiled function
        return self.fft_func(problem)