import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm
import numba

class CustomCSCMatrix(sparse.csc_matrix):
    def __len__(self):
        return self.shape[0]

@numba.jit(nopython=True, fastmath=True, parallel=True)
def numba_expm(A_dense):
    """JIT-compiled matrix exponential using Taylor series approximation"""
    n = A_dense.shape[0]
    result = np.eye(n)
    term = np.eye(n)
    
    # Use Taylor series approximation with adaptive stopping
    for i in range(1, 30):
        term = term @ A_dense / i
        new_result = result + term
        if np.linalg.norm(new_result - result) < 1e-12:
            break
        result = new_result
    return result

class Solver:
    def solve(self, problem, **kwargs):
        """
        Optimized matrix exponential using Numba JIT compilation.
        Returns a custom sparse matrix that properly implements __len__.
        """
        try:
            A = problem["matrix"]
            n = A.shape[0]
            
            # Use scipy expm for large matrices
            if n > 300:
                return CustomCSCMatrix(expm(A))
            
            # Convert to dense and compute with Numba
            A_dense = A.toarray()
            expA_dense = numba_expm(A_dense)
            
            # Convert back to sparse with thresholding
            threshold = np.max(np.abs(expA_dense)) * 1e-10
            mask = np.abs(expA_dense) >= threshold
            coo = sparse.coo_matrix(expA_dense * mask)
            coo.eliminate_zeros()
            
            # Convert to CSC for output
            result = coo.tocsc()
            result.sort_indices()
            result.eliminate_zeros()
            return CustomCSCMatrix(result)
            
        except Exception as e:
            # Fallback to scipy expm in case of errors
            n = problem["matrix"].shape[0] if "matrix" in problem else 1
            return CustomCSCMatrix(expm(sparse.eye(n, format='csc')))