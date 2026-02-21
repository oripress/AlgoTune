import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

@jit
def fast_matmul(A, B):
    return jax.lax.dot(A, B)

class Solver:
    def __init__(self):
        fast_matmul(jnp.zeros((2, 2)), jnp.zeros((2, 2)))

    def solve(self, problem: dict[str, list[list[float]]], **kwargs):
        A = jax.device_put(problem["A"])
        B = jax.device_put(problem["B"])
        return fast_matmul(A, B)