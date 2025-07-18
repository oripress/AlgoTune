import numpy as np
import scipy.linalg
import numba
from typing import Any

# Use Numba to create a highly optimized function to check for triangularity.
# This is faster than NumPy equivalents as it avoids intermediate arrays and short-circuits.
@numba.njit(cache=True)
def is_triangular(A: np.ndarray) -> bool:
    """Checks if a matrix is upper or lower triangular."""
    n = A.shape[0]
    # Check for upper triangular (all elements below diagonal are zero)
    is_upper = True
    for i in range(1, n):
        for j in range(i):
            if A[i, j] != 0.0:
                is_upper = False
                break
        if not is_upper:
            break
    if is_upper:
        return True

    # Check for lower triangular (all elements above diagonal are zero)
    is_lower = True
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0.0:
                is_lower = False
                break
        if not is_lower:
            break
    return is_lower

# Use Numba for a fast symmetry check that mimics np.allclose and short-circuits.
@numba.njit(cache=True)
def is_symmetric_numba(A: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Checks if a matrix is symmetric within a given tolerance."""
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            # Emulate np.allclose logic: |a - b| <= atol + rtol * |b|
            diff = abs(A[i, j] - A[j, i])
            if diff > (atol + rtol * abs(A[j, i])):
                return False
    return True

class Solver:
    def solve(self, problem: list[list[float]], **kwargs) -> Any:
        """
        Solve the eigenvalue problem with a Numba-accelerated hierarchical optimization.
        All checks (triangular, symmetric) are JIT-compiled for minimal overhead.
        """
        problem_arr = np.array(problem, dtype=np.float64)
        
        if problem_arr.size == 0:
            return []
        
        # Optimization 1: Fast triangular check using Numba.
        if is_triangular(problem_arr):
            eigenvalues = np.diag(problem_arr)
            sorted_eigenvalues = -np.sort(-eigenvalues)
            return sorted_eigenvalues.astype(np.complex128).tolist()

        # Optimization 2: Fast symmetric check using Numba.
        if is_symmetric_numba(problem_arr):
            eigenvalues = scipy.linalg.eigh(problem_arr, eigvals_only=True)
            sorted_eigenvalues = np.flip(eigenvalues)
            return sorted_eigenvalues.astype(np.complex128).tolist()
        
        # General case for dense, non-symmetric matrices.
        eigenvalues = scipy.linalg.eigvals(problem_arr)
        sorted_eigenvalues = -np.sort(-eigenvalues)
        return sorted_eigenvalues.tolist()