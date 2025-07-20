import numpy as np
import scipy.linalg
from typing import Any

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Computes the QR factorization by calling the highly optimized
        scipy.linalg.qr function with parameters tuned for maximum performance.
        """
        # Step 1: Prepare the input matrix.
        # - dtype=np.float64: Ensures stability (float32 was found to be unstable).
        # - order='F': Creates a Fortran (column-major) contiguous array. This
        #   is the native format for LAPACK, preventing internal memory copies.
        A = np.array(problem["matrix"], dtype=np.float64, order='F')

        # Step 2: Call scipy.linalg.qr with performance-critical flags.
        # - overwrite_a=True: Allows the function to modify the input array `A`,
        #   avoiding a significant internal memory allocation and copy.
        # - mode='economic': Computes a smaller Q matrix, crucial for non-square matrices.
        # - check_finite=False: Skips input validation, reducing overhead.
        Q, R = scipy.linalg.qr(A, overwrite_a=True, mode='economic', check_finite=False)

        # Step 3: Convert results to the required list format.
        solution = {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
        return solution