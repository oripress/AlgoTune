import numpy as np
from scipy.sparse.linalg import eigsh
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, List[List[float]]], **kwargs) -> Any:
        """
        Finds the two eigenvalues of a symmetric matrix closest to zero.

        This final implementation uses the most performant and robust method identified
        through extensive testing: `scipy.sparse.linalg.eigsh` in shift-and-invert mode.

        The strategy is as follows:
        1.  The `eigsh` function is an iterative solver based on the ARPACK library,
            designed to find a few eigenvalues of a large matrix efficiently.
        2.  By setting `sigma=0` and `which='LM'`, we use the "shift-and-invert"
            strategy. This transforms the problem to find the eigenvalues of
            (A - 0*I)^-1 with the largest magnitude. These correspond precisely
            to the eigenvalues of the original matrix A that are closest to 0.
        3.  This approach is significantly faster than the baseline `numpy.linalg.eigvalsh`,
            which computes all N eigenvalues.
        4.  A `try...except` block provides crucial robustness. The shift-and-invert
            method involves an internal matrix inversion, which can fail if the
            matrix is singular. In such cases, the code falls back to the slower
            but guaranteed numpy method.
        5.  A direct conditional check is used to sort the two resulting eigenvalues
            by magnitude, which is a micro-optimization over a general sort.
        """
        matrix = np.array(problem["matrix"], dtype=np.float64)

        try:
            # Use the shift-and-invert strategy to find the 2 eigenvalues closest to 0.
            eigenvalues = eigsh(
                matrix,
                k=2,
                sigma=0,
                which='LM',
                return_eigenvectors=False,
            )
            # For two elements, a direct comparison is faster than sorted().
            if abs(eigenvalues[0]) > abs(eigenvalues[1]):
                return [eigenvalues[1], eigenvalues[0]]
            else:
                return eigenvalues.tolist()
        except Exception:
            # Fallback for cases where eigsh fails (e.g., singular matrix).
            # This computes all eigenvalues and is slower but robust.
            eigenvalues = np.linalg.eigvalsh(matrix)
            # Use numpy's efficient sorting to find the two smallest by magnitude.
            indices = np.argsort(np.abs(eigenvalues))
            return eigenvalues[indices[:2]].tolist()