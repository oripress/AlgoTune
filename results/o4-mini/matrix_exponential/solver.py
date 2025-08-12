import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem, **kwargs):
        # Convert input to NumPy array
        A = np.asarray(problem["matrix"], dtype=float)
        # Compute the matrix exponential using SciPy's optimized routine
        expA = expm(A)
        # Return result as nested Python lists for validation
        return {"exponential": expA.tolist()}