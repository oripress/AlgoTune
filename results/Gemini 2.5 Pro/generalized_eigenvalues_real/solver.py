import numpy as np
from scipy.linalg import eigh
from typing import Any

class Solver:
    # Define a threshold for switching between LAPACK drivers. The performance
    # of eigenvalue algorithms is size-dependent.
    # - 'gv' (standard QR algorithm) has lower overhead and is faster for smaller matrices.
    # - 'gvd' (divide-and-conquer) has better asymptotic complexity but higher
    #   overhead, making it faster for larger matrices.
    # A threshold around 250 is a common heuristic based on LAPACK performance guides.
    GVD_THRESHOLD = 250

    def solve(self, problem: Any, **kwargs) -> Any:
        """
        Solves the generalized eigenvalue problem by adaptively selecting the
        optimal LAPACK driver based on matrix size.
        """
        A, B = problem
        N = A.shape[0]

        # Adaptively choose the driver based on the matrix size.
        if N < self.GVD_THRESHOLD:
            # For smaller matrices, the standard driver is typically faster.
            driver = 'gv'
        else:
            # For larger matrices, the divide-and-conquer driver is superior.
            driver = 'gvd'

        # Call the highly optimized eigh function with the selected driver and
        # performance-enhancing flags.
        # - overwrite_a/b=True: Avoids internal copies by allowing in-place modification.
        # - check_finite=False: Skips validation checks for a small speed gain.
        eigenvalues = eigh(A, B, eigvals_only=True, driver=driver,
                           overwrite_a=True, overwrite_b=True, check_finite=False)

        # The eigenvalues are returned in ascending order. Flip the array to meet
        # the problem's descending order requirement and convert to a list.
        return np.flip(eigenvalues).tolist()