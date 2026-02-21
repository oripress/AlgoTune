import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import expm
import numpy as np

class Solver:
    def __init__(self):
        self.expm_jit = jax.jit(expm)
        
    def solve(self, problem: dict, **kwargs) -> dict:
        return {"exponential": np.asarray(self.expm_jit(problem["matrix"]))}