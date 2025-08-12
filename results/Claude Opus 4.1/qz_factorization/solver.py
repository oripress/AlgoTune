import numpy as np
from scipy.linalg import qz

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the QZ factorization problem by computing the QZ factorization of (A,B).
        """
        # Use asarray for potentially faster conversion if input is already array-like
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        
        # Use scipy's qz with overwrite flags to save memory
        AA, BB, Q, Z = qz(A, B, output="real", overwrite_a=False, overwrite_b=False)
        
        # Direct dictionary construction without intermediate variables
        return {
            "QZ": {
                "AA": AA.tolist(),
                "BB": BB.tolist(),
                "Q": Q.tolist(),
                "Z": Z.tolist()
            }
        }