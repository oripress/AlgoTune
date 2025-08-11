import numpy as np
import jax.numpy as jnp
import scipy.fft

class Solver:
    def solve(self, problem):
        """
        Compute the N-dimensional DST Type II using scipy.fft.
        """
        # Convert to jax array for potential GPU acceleration
        problem_jax = jnp.asarray(problem)
        
        # Use scipy's optimized DST-II implementation
        result = scipy.fft.dstn(np.asarray(problem_jax), type=2)
        return result