import numpy as np
from scipy.linalg import get_lapack_funcs
from typing import Any

# Fetch the LAPACK function handle once at module load time. This avoids the
# get_lapack_funcs overhead in the timed solve method. The function is
# retrieved for float64 arrays by using a dummy array.
_POTRF, = get_lapack_funcs(('potrf',), (np.empty((1, 1), dtype=np.float64),))

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Computes the Cholesky factorization using a maximally optimized approach.
        
        The key insight is that the standard `np.array(matrix)` conversion is
        suboptimal because it processes the entire N*N matrix, while the Cholesky
        algorithm only requires the lower triangle for a symmetric matrix.
        
        The strategy is:
        1. Manually create an empty N*N NumPy array.
        2. Loop through the input list of lists and copy *only* the lower-triangular
           elements. This avoids converting the ~N^2/2 redundant elements in the
           upper triangle and is significantly faster than `np.array(matrix)`.
        3. Pass this array to a pre-fetched LAPACK `potrf` function.
           - The function is fetched once at module load to avoid lookup overhead.
           - `lower=True` ensures it only reads the lower triangle we copied.
           - `overwrite_a=True` performs the computation in-place.
           - `clean=True` efficiently zeroes out the uninitialized upper triangle
             of the array at the C level.
        """
        matrix = problem["matrix"]
        n = len(matrix)
        
        # Create an uninitialized C-contiguous array. This is faster than np.zeros.
        A = np.empty((n, n), dtype=np.float64)
        
        # Manually copy only the lower triangle from the input list of lists.
        # This is the critical optimization, as it avoids processing the
        # redundant upper-triangular elements.
        for i, row in enumerate(matrix):
            A[i, :i+1] = row[:i+1]
            
        # Call the pre-fetched LAPACK function directly with all optimizations enabled.
        L, info = _POTRF(a=A, lower=True, clean=True, overwrite_a=True)
        
        # The problem guarantees positive-definite matrices, so info will be 0.
        
        return {"Cholesky": {"L": L}}