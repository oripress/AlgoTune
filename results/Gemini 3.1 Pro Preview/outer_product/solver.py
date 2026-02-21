import numpy as np
import jax.numpy as jnp
import jax

@jax.jit
def jax_outer(a, b):
    return jnp.outer(a, b)

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        a, b = problem
        return np.asarray(jax_outer(a, b))