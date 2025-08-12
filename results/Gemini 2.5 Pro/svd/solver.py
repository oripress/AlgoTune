import numpy as np
from scipy.linalg import svd
from typing import Any

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Computes the SVD of a matrix using scipy.linalg.svd, which can be
        faster than the numpy equivalent by leveraging performance-tuning flags.
        """
        # Convert the input matrix to a NumPy array with double precision.
        A = np.array(problem["matrix"], dtype=np.float64)

        # Use scipy's SVD implementation with performance optimizations:
        # - full_matrices=False: Computes the "thin" SVD, which is faster and
        #   more memory-efficient.
        # - overwrite_a=True: Allows the function to modify the input array 'A'
        #   in place, avoiding an internal copy and saving time/memory. This is
        #   safe as 'A' is created locally within this function.
        # - check_finite=False: Skips checking for NaNs/Infs in the input,
        #   providing a small speed boost. We assume valid inputs.
        U, s, Vh = svd(A, full_matrices=False, overwrite_a=True, check_finite=False)

        # The svd function returns V transpose (Vh), so we transpose it to get V.
        V = Vh.T
        
        solution = {
            "U": U,
            "S": s,
            "V": V
        }
        return solution