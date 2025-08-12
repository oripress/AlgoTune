import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the orthogonal Procrustes problem with optimized memory layout.
        """
        A = problem.get("A")
        B = problem.get("B")
        
        if A is None or B is None:
            return {}
        
        # Ensure contiguous memory layout for optimal BLAS performance
        A = np.ascontiguousarray(A, dtype=np.float64)
        B = np.ascontiguousarray(B, dtype=np.float64)
        
        if A.shape != B.shape:
            return {}
        
        # Compute M = B @ A^T using optimized GEMM
        M = np.dot(B, A.T, out=None)
        
        # Use numpy's optimized SVD (automatically uses LAPACK)
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        
        # Compute G = U @ V^T
        G = np.dot(U, Vt, out=None)
        
        return {"solution": G.tolist()}