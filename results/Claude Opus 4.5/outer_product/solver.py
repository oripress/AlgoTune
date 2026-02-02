import numpy as np
import jax.numpy as jnp
from jax import jit
import jax

jax.config.update('jax_platform_name', 'cpu')

@jit
def jax_outer(v1, v2):
    return jnp.outer(v1, v2)

class Solver:
    def __init__(self):
        # Warm up JAX JIT
        dummy = jnp.array([1.0, 2.0])
        _ = jax_outer(dummy, dummy).block_until_ready()
    
    def solve(self, problem, **kwargs):
        vec1, vec2 = problem
        # Convert to jax arrays and compute
        v1 = jnp.asarray(vec1)
        v2 = jnp.asarray(vec2)
        result = jax_outer(v1, v2)
        return np.asarray(result)