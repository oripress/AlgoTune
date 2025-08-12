import numpy as np
from scipy.linalg import svd

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """
        Solve the orthogonal Procrustes problem: find orthogonal matrix G 
        that minimizes ||GA - B||_F^2 subject to G^T G = I.
        
        The solution is obtained by:
        1. Computing M = B @ A.T
        2. Computing SVD: M = U @ S @ Vt
        3. The optimal orthogonal matrix is G = U @ Vt
        
        :param problem: Dictionary with keys "A" and "B" containing n x n matrices
        :return: Dictionary with key "solution" containing the optimal orthogonal matrix G
        """
        A = problem.get("A")
        B = problem.get("B")
        
        if A is None or B is None:
            return {}
            
        A = np.array(A, dtype=np.float64, order='C')
        B = np.array(B, dtype=np.float64, order='C')
        
        if A.shape != B.shape:
            return {}
            
        # Compute M = B @ A.T
        M = np.dot(B, A.T)
        
        # Compute SVD using scipy with gesdd driver (faster than numpy's default)
        U, S, Vt = svd(M, full_matrices=False, lapack_driver='gesdd')
        
        # Compute G = U @ Vt
        G = np.dot(U, Vt)
        
        return {"solution": G.tolist()}