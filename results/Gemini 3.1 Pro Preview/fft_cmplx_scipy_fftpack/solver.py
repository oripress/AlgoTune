import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

def jax_fftn(x):
    return jnp.fft.fftn(x)
jax_fftn = jax.jit(jax_fftn, backend='cpu', inline=True)

class Solver:
    def __init__(self):
        # Precompile for common square matrix sizes
        for n in [16, 32, 64, 128, 256, 512, 1024]:
            dummy = jnp.zeros((n, n), dtype=jnp.complex128)
            jax_fftn(dummy)

    def solve(self, problem, **kwargs) -> Any:
        return np.asarray(jax_fftn(problem))