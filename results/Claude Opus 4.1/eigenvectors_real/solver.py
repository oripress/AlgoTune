import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, List
import numba as nb

@nb.jit(nopython=True, cache=True, fastmath=True)
def convert_eigenvectors_to_list(eigenvectors, n):
    """Fast conversion of eigenvectors to list of lists using Numba."""
    result = []
    for i in range(n):
        vec = []
        for j in range(n):
            vec.append(eigenvectors[j, i])
        result.append(vec)
    return result

class Solver:
    def __init__(self):
        # Pre-compile the numba function
        dummy = np.array([[1.0, 0.0], [0.0, 1.0]])
        _ = convert_eigenvectors_to_list(dummy, 2)
    
    def solve(self, problem: NDArray) -> Tuple[List[float], List[List[float]]]:
        """
        Solve the eigenvalue problem for the given real symmetric matrix.
        Optimized with Numba JIT compilation.
        """
        n = problem.shape[0]
        
        # Use numpy's eigh which calls optimized LAPACK routines
        eigenvalues, eigenvectors = np.linalg.eigh(problem)
        
        # Reverse for descending order (in-place for efficiency)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # Use Numba-compiled function for fast list conversion
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = convert_eigenvectors_to_list(eigenvectors, n)
        
        return (eigenvalues_list, eigenvectors_list)