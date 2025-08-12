import numpy as np
from scipy.linalg import qz

class Solver:
    def solve(self, problem, **kwargs):
        # Create arrays with minimal overhead using array constructor
        # Use memory view for faster access
        A = np.array(problem["A"], dtype=np.float64, order='F')
        B = np.array(problem["B"], dtype=np.float64, order='F')
        
        # Perform QZ decomposition with optimized parameters
        AA, BB, Q, Z = qz(A, B, output='real', check_finite=False)
        
        # Return results directly without intermediate variables
        return {"QZ": {"AA": AA.tolist(), "BB": BB.tolist(), "Q": Q.tolist(), "Z": Z.tolist()}}