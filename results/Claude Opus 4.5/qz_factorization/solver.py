import numpy as np
from scipy.linalg import qz
import jax.numpy as jnp
from jax.scipy.linalg import qr

class Solver:
    def __init__(self):
        # Pre-import to avoid first-call overhead
        pass
        
    def solve(self, problem, **kwargs):
        A = np.array(problem["A"], dtype=np.float64, order='F')
        B = np.array(problem["B"], dtype=np.float64, order='F')
        
        # Use scipy which is highly optimized for this specific operation
        AA, BB, Q, Z = qz(A, B, output="real", overwrite_a=True, overwrite_b=True, check_finite=False)
        
        # tolist() is the standard approach
        return {"QZ": {"AA": AA.tolist(), "BB": BB.tolist(), "Q": Q.tolist(), "Z": Z.tolist()}}