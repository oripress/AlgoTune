import numpy as np
import jax.numpy as jnp
from jax import jit
from scipy.linalg import qz

class Solver:
    def __init__(self):
        # Pre-compile the solve function
        self._jit_solve = jit(self._solve_jax)
    
    def _solve_jax(self, A, B):
        # Use scipy's QZ but with JAX arrays for potential optimization
        return qz(A, B, output="real")
    
    def solve(self, problem: dict[str, list[list[float]]]) -> dict[str, dict[str, list[list[float | complex]]]]:
        """
        Solve the QZ factorization problem by computing the QZ factorization of (A,B).
        """
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        
        # Direct scipy call is still fastest for this algorithm
        AA, BB, Q, Z = qz(A, B, output="real")
        
        solution = {"QZ": {"AA": AA.tolist(), "BB": BB.tolist(), "Q": Q.tolist(), "Z": Z.tolist()}}
        return solution