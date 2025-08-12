import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the orthogonal Procrustes problem.
        Given matrices A and B, find orthogonal matrix G that minimizes ||GA - B||_F^2
        """
        A = problem.get("A")
        B = problem.get("B")
        
        if A is None or B is None:
            return {}
        
        A = np.array(A)
        B = np.array(B)
        
        if A.shape != B.shape:
            return {}
        
        # Compute M = B @ A.T
        M = B @ A.T
        
        # Compute SVD of M
        U, _, Vt = np.linalg.svd(M)
        
        # Compute G = U @ Vt
        G = U @ Vt
        
        return {"solution": G.tolist()}