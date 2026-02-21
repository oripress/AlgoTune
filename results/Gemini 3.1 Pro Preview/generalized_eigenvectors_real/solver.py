import scipy.linalg
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        A, B = problem
        
        # Use scipy.linalg.eigh with the sygvd driver which is optimized for this problem
        # overwrite_a and overwrite_b allow scipy to use the input arrays as workspace
        # check_finite=False skips the NaN/Inf checks for speed
        w, v = scipy.linalg.eigh(A, B, driver='gvd', overwrite_a=True, overwrite_b=True, check_finite=False)
        
        # Reverse to get descending order and convert to lists
        # v.T makes eigenvectors rows, [::-1] reverses the rows (which corresponds to reversing columns of v)
        return w[::-1].tolist(), v.T[::-1].tolist()