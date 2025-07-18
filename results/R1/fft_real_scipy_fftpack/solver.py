import jax
import jax.numpy as jnp
import numpy as np
import os
from functools import lru_cache

# Configure environment for maximum performance
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_DYNAMIC'] = 'FALSE'

# Configure JAX for optimal CPU performance
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_default_matmul_precision', 'highest')

@lru_cache(maxsize=20)
def get_jax_fft_fn(shape, dtype_name):
    """Create JIT-compiled JAX FFT function with LRU cache"""
    dtype = getattr(np, dtype_name)
    
    @jax.jit
    def jax_fft(x):
        return jnp.fft.fftn(x)
    
    # Pre-compile
    dummy = np.zeros(shape, dtype=dtype)
    jax_fft(jax.device_put(dummy)).block_until_ready()
    return jax_fft

class Solver:
    def __init__(self):
        # Pre-warm FFT for common sizes
        sizes = [32, 64, 128, 256, 512, 1024, 2048]
        dtypes = ['float32', 'float64']
        
        for size in sizes:
            for dtype_name in dtypes:
                get_jax_fft_fn((size, size), dtype_name)
    
    def solve(self, problem):
        shape = problem.shape
        dtype_name = problem.dtype.name
        
        # Get compiled FFT function
        fft_fn = get_jax_fft_fn(shape, dtype_name)
        
        # Transfer data with zero-copy if possible
        if not isinstance(problem, jax.Array):
            problem = jax.device_put(problem)
        
        # Execute FFT
        result = fft_fn(problem)
        
        # Block until ready for large matrices
        if shape[0] >= 1024:  # Higher threshold to reduce overhead
            result.block_until_ready()
            
        return np.array(result, copy=False)