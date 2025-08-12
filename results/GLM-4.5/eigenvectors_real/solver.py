import jax.numpy as jnp
from jax.scipy.linalg import eigh
import jax

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

class Solver:
    def solve(self, problem, **kwargs) -> tuple[list[float], list[list[float]]]:
        # Convert input to JAX array
        problem_jax = jnp.array(problem)
        
        # Use JAX's eigh function
        eigenvalues, eigenvectors = eigh(problem_jax)
        
        # Reverse to get descending order
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # Convert back to regular numpy arrays then to lists
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = eigenvectors.T.tolist()
        
        return (eigenvalues_list, eigenvectors_list)