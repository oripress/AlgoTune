from __future__ import annotations
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, LinearOperator, aslinearoperator
import warnings

# Try to import the compiled Cython module
try:
    import solver_cython
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        # Get matrix data and k
        matrix_data = problem["matrix"]
        k: int = int(problem["k"])
        
        # Check if matrix is already sparse
        if sparse.issparse(matrix_data):
            mat = matrix_data.asformat("csr")
            n = mat.shape[0]
        else:
            # It's a list of lists, check sparsity
            n = len(matrix_data)
            if n == 0:
                return []
            
            total_elements = n * n
            # Use Cython for faster zero counting if available
            if HAS_CYTHON:
                zero_count = solver_cython.count_zeros_cython(matrix_data)
            else:
                zero_count = sum(row.count(0) for row in matrix_data)
            sparsity = zero_count / total_elements
            
            # Use CSR format for sparse matrices, otherwise use dense directly
            if sparsity > 0.5:  # More than 50% zeros
                # Try different sparse formats for optimal performance
                mat = sparse.csr_matrix(matrix_data)
                # Also try CSC format which might be better for some operations
                mat_csc = mat.tocsc()
                # Use the format with fewer non-zero elements
                if mat_csc.nnz < mat.nnz:
                    mat = mat_csc
            else:
                # For dense matrices, work directly with numpy array
                mat_array = np.array(matrix_data, dtype=np.float64)
                # Use Cython for faster dense computation if available
                if HAS_CYTHON:
                    vals = solver_cython.fast_eigvalsh_fallback(mat_array, k)
                else:
                    vals = np.linalg.eigvalsh(mat_array)
                return [float(v) for v in vals[:k]]
        
        # Early return for edge cases
        if k == 0:
            return []
        if k >= n or n < 2 * k + 1:
            mat_array = mat.toarray()
            # Use Cython for faster dense computation if available
            if HAS_CYTHON:
                vals = solver_cython.fast_eigvalsh_fallback(mat_array, k)
            else:
                vals = np.linalg.eigvalsh(mat_array)
            return [float(v) for v in vals[:k]]
        
        # Suppress warnings for faster execution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Optimized sparse eigenvalue computation
            try:
                # Calculate sparsity ratio for intelligent parameter selection
                sparsity_ratio = 1.0 - (mat.nnz / (n * n))
                
                # Ultra-optimized parameters based on matrix size and sparsity
                if n > 300:
                    # Very large matrices - be aggressive but safe
                    ncv = min(n - 1, max(4 * k + 1, 40))
                    maxiter = n * 20
                    tol = 1e-4 if sparsity_ratio > 0.85 else 1e-5
                elif n > 150:
                    ncv = min(n - 1, max(3 * k + 1, 35))
                    maxiter = n * 35
                    tol = 1e-5 if sparsity_ratio > 0.75 else 1e-6
                elif n > 75:
                    ncv = min(n - 1, max(3 * k + 1, 30))
                    maxiter = n * 50
                    tol = 1e-6
                else:
                    ncv = min(n - 1, max(2 * k + 1, 25))
                    maxiter = n * 100
                    tol = 1e-7
                
                # Choose optimal matrix format based on sparsity
                if sparsity_ratio > 0.7 and sparse.isspmatrix_csc(mat):
                    # CSC might be better for very sparse matrices
                    mat_optimal = mat
                elif sparsity_ratio > 0.7:
                    # Convert to CSC for very sparse matrices
                    mat_optimal = mat.tocsc()
                else:
                    # Use CSR for moderately sparse matrices
                    mat_optimal = mat
                
                vals = eigsh(
                    mat_optimal,
                    k=k,
                    which="SM",
                    return_eigenvectors=False,
                    maxiter=maxiter,
                    ncv=ncv,
                    tol=tol,
                    v0=None
                )
                
            except Exception:
                try:
                    # Fallback with LinearOperator
                    def matvec(x):
                        return mat @ x
                    
                    lin_op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
                    
                    vals = eigsh(
                        lin_op,
                        k=k,
                        which="SM",
                        return_eigenvectors=False,
                        maxiter=n * 100,
                        ncv=min(n - 1, max(2 * k + 1, 20)),
                        tol=1e-6
                    )
                    
                except Exception:
                    # Last-resort dense fallback
                    mat_array = mat.toarray()
                    # Use Cython for faster dense computation if available
                    if HAS_CYTHON:
                        vals = solver_cython.fast_eigvalsh_fallback(mat_array, k)
                    else:
                        vals = np.linalg.eigvalsh(mat_array)
        
        return [float(v) for v in np.sort(np.real(vals))]