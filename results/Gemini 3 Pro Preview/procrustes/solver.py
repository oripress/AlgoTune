import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        # Load A and B. np.array infers dtype, usually float64 for python floats.
        # We keep them in float64 to compute M accurately.
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        
        # M = B @ A.T (float64)
        M = B @ A.T
        
        # Perform SVD in float32 for speed.
        # The polar factor G = U @ Vt is generally well-conditioned.
        U, _, Vt = np.linalg.svd(M.astype(np.float32))
        
        # G = U @ Vt (float32)
        G = U @ Vt
        
        # Return as float64 list
        return {"solution": G.astype(np.float64).tolist()}