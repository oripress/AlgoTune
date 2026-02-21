import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

@jax.jit
def jax_solve(A, B):
    M = jnp.dot(B, A.T)
    U, S, Vt = jnp.linalg.svd(M, full_matrices=False)
    return jnp.dot(U, Vt)

class Solver:
    def __init__(self):
        # Warm up the JIT compiler with a few common sizes
        for n in [2, 3, 4, 5, 10, 50, 100]:
            dummy_A = jnp.zeros((n, n), dtype=jnp.float64)
            dummy_B = jnp.zeros((n, n), dtype=jnp.float64)
            jax_solve(dummy_A, dummy_B)

    def solve(self, problem: dict, **kwargs) -> dict:
        # Best performing solution using JAX
        A = jnp.array(problem["A"], dtype=jnp.float64)
        B = jnp.array(problem["B"], dtype=jnp.float64)
        
        G = jax_solve(A, B)
        
        return {"solution": np.array(G).tolist()}