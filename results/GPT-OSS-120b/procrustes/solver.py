import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Orthogonal Procrustes solver.
        Given matrices A and B (both n×n), returns the orthogonal matrix G
        that minimizes ||G @ A - B||_F.
        """
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None:
            return {}
        # Convert inputs to float64 arrays for numerical accuracy
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        if A.shape != B.shape:
            return {}
        # Compute M = B @ A.T and its SVD
        U, _, Vt = np.linalg.svd(B @ A.T, full_matrices=False)
        # Orthogonal solution G = U @ Vt
        G = U @ Vt
        # Return as nested list for JSON compatibility
        return {"solution": G.tolist()}
        return {"solution": G.tolist()}