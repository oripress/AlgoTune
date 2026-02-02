import numpy as np
import fast_solver
from scipy.linalg import blas

class Solver:
    def solve(self, problem: dict[str, list[list[float]]], **kwargs) -> np.ndarray:
        A, B = fast_solver.convert_to_arrays(problem["A"], problem["B"])
        
        # Ensure arrays are contiguous and float64 for dgemm
        # fast_solver creates C-contiguous arrays of float64
        
        # dgemm(alpha, a, b, beta, c, trans_a, trans_b, overwrite_c)
        # C = alpha * op(A) * op(B) + beta * C
        # Default is C = 1.0 * A * B
        
        # Note: scipy.linalg.blas.dgemm expects Fortran-contiguous arrays for best performance?
        # Or it handles C-contiguous by transposing?
        # Actually, numpy.dot is very optimized. Let's see if dgemm is faster.
        # If A and B are C-contiguous, dgemm might treat them as transposed F-contiguous.
        # A (n x m) C-contig == A.T (m x n) F-contig
        # B (m x p) C-contig == B.T (p x m) F-contig
        # We want C = A * B.
        # If we pass A and B to dgemm, it sees A.T and B.T.
        # It computes C_f = A.T * B.T = (B * A).T
        # So the result C_f is (p x n) F-contig, which is (n x p) C-contig.
        # But wait, (B*A) is not A*B.
        
        # Let's stick to np.dot first, but use the fast conversion.
        return np.dot(A, B)