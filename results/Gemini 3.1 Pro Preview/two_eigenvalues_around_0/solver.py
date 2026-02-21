import numpy as np
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[float]:
        matrix = np.array(problem["matrix"], dtype=float)
        n = matrix.shape[0]
        if n <= 10:
            eigenvalues = np.linalg.eigvalsh(matrix)
        else:
            try:
                eigenvalues = eigsh(matrix, k=2, sigma=0, tol=1e-3, ncv=5, v0=np.ones(n), return_eigenvectors=False)
            except Exception:
                eigenvalues = np.linalg.eigvalsh(matrix)
                
        eigenvalues_sorted = sorted(eigenvalues, key=abs)
        return [float(x) for x in eigenvalues_sorted[:2]]