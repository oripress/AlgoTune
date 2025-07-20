import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import ArpackError

class Solver:
    def solve(self, problem, **kwargs):
        matrix = np.array(problem["matrix"], dtype=float)
        size = len(matrix)
        
        # For small matrices, use direct method
        if size <= 100:
            eigenvalues = np.linalg.eigvalsh(matrix)
            eigenvalues_sorted = sorted(eigenvalues, key=abs)[:2]
            return [float(x) for x in eigenvalues_sorted]
        else:
            # For larger matrices, use iterative method targeting near zero
            try:
                # Optimize parameters based on matrix size
                ncv = min(max(20, int(size**0.5)), size)
                tol = 1e-5 if size < 1000 else 1e-4
                maxiter = 1000 if size < 5000 else 500
                
                # Use shift-invert mode for stability near zero
                evals = eigsh(
                    matrix,
                    k=2,
                    sigma=0,
                    which='LM',
                    mode='normal',
                    return_eigenvectors=False,
                    tol=tol,
                    maxiter=maxiter,
                    ncv=ncv
                )
                # Ensure eigenvalues are real
                evals = np.real(evals)
                evals_sorted = sorted(evals, key=abs)
                return [float(x) for x in evals_sorted]
            except ArpackError:
                # Fallback to direct method if iterative fails
                eigenvalues = np.linalg.eigvalsh(matrix)
                eigenvalues_sorted = sorted(eigenvalues, key=abs)[:2]
                return [float(x) for x in eigenvalues_sorted]