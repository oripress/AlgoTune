import numpy as np
from scipy.linalg import expm as scipy_expm
from typing import Any

class Solver:
    def __init__(self):
        """
        Initializes the solver. This strategy requires no pre-computation or caching
        beyond what the underlying libraries (NumPy/SciPy) already do.
        """
        pass

    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Computes the matrix exponential using a hybrid strategy to outperform
        the standard Scipy implementation on average.

        The strategy is as follows:
        1. Check for diagonal matrices: A very fast O(n) path for computation
           after an O(n^2) check. This is the simplest and fastest case.
        2. Check for symmetric matrices: An O(n^3) path using eigendecomposition.
           This is typically faster than the general expm algorithm for
           symmetric matrices because it can use specialized, highly optimized
           routines (`numpy.linalg.eigh`).
        3. Fallback: For all other matrices, use the highly optimized
           `scipy.linalg.expm`.

        This approach aims to achieve a better average performance by identifying
        common special matrix types and solving them with more efficient,
        specialized algorithms. The overhead of the checks is negligible (O(n^2))
        compared to the O(n^3) cost of the matrix exponential itself.
        """
        A_list = problem["matrix"]
        A = np.array(A_list, dtype=np.float64)

        # Path 1: Diagonal matrix (check is O(n^2), computation is O(n))
        if np.count_nonzero(A - np.diag(np.diag(A))) == 0:
            exp_diag = np.exp(np.diag(A))
            expA = np.diag(exp_diag)
            return {"exponential": expA.tolist()}

        # Path 2: Symmetric matrix (check is O(n^2), computation is O(n^3) via eigh)
        # This is often faster than general-purpose expm for symmetric matrices.
        if np.allclose(A, A.T):
            # For a symmetric matrix A, exp(A) = V * exp(D) * V.T, where A = V * D * V.T
            # is the eigendecomposition. `np.linalg.eigh` is optimized for this.
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            exp_eigenvalues = np.exp(eigenvalues)
            
            # Reconstruct the matrix: exp(A) = V @ diag(exp(w)) @ V.T
            # This is an efficient way to compute V @ diag(d) @ V.T in numpy
            expA = (eigenvectors * exp_eigenvalues) @ eigenvectors.T
            return {"exponential": expA.tolist()}

        # Path 3: Fallback for general matrices
        # Use the robust and optimized baseline implementation. The performance gain
        # comes from diverting symmetric/diagonal matrices to faster paths.
        expA = scipy_expm(A)
        return {"exponential": expA.tolist()}