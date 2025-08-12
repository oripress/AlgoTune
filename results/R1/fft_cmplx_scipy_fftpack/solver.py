import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
from jax import jit

# Precompile JIT for common sizes
FFT_FUNCS = {}
COMMON_SIZES = [32, 64, 128, 256, 512, 1024]  # Extended square matrix sizes

for size in COMMON_SIZES:
    shape = (size, size)
    
    @jit
    def fft_func(x):
        return jnp.fft.fftn(x)
    
    # Warm up the JIT
    dummy = np.zeros(shape, dtype=np.complex128)
    fft_func(dummy).block_until_ready()
    FFT_FUNCS[shape] = fft_func

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """Optimized FFT with precompiled functions and async transfers."""
        shape = problem.shape
        
        # Check if input is already in optimal format
        if problem.flags['C_CONTIGUOUS'] and problem.dtype == np.complex128:
            arr = problem
        else:
            arr = np.asarray(problem, dtype=np.complex128, order='C')
        
        # Asynchronous device transfer
        jax_arr = jax.device_put(arr)
        
        if shape in FFT_FUNCS:
            result = FFT_FUNCS[shape](jax_arr)
        else:
            # Use JAX's FFT with JIT compilation
            result = jnp.fft.fftn(jax_arr)
        
        # Block until result is ready
        result.block_until_ready()
        return np.asarray(result)