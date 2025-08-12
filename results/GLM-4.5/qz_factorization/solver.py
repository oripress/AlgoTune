import numpy as np
from scipy.linalg import qz
import jax
import jax.numpy as jnp

class Solver:
    def solve(self, problem, **kwargs):
        # Convert input to numpy arrays first
        A_np = np.array(problem["A"])
        B_np = np.array(problem["B"])
        
        # Check if JAX can use GPU/TPU
        try:
            # Determine data type
            dtype = complex if np.iscomplexobj(A_np) or np.iscomplexobj(B_np) else float
            
            # Convert to JAX arrays
            A = jnp.array(A_np, dtype=dtype)
            B = jnp.array(B_np, dtype=dtype)
            
            # Use JAX's QZ factorization
            AA, BB, Q, Z = jax.linalg.qz(A, B)
            
            # Convert results back to numpy
            AA_np = np.array(AA)
            BB_np = np.array(BB)
            Q_np = np.array(Q)
            Z_np = np.array(Z)
        except Exception:
            # Fall back to SciPy if JAX fails
            if np.iscomplexobj(A_np) or np.iscomplexobj(B_np):
                AA_np, BB_np, Q_np, Z_np = qz(A_np, B_np, output='complex')
            else:
                AA_np, BB_np, Q_np, Z_np = qz(A_np, B_np, output='real')
        
        # Convert results to lists and return
        solution = {
            "QZ": {
                "AA": AA_np.tolist(),
                "BB": BB_np.tolist(),
                "Q": Q_np.tolist(),
                "Z": Z_np.tolist()
            }
        }
        return solution