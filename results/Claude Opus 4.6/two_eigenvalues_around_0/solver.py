import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh as scipy_eigvalsh

class Solver:
    def solve(self, problem, **kwargs):
        matrix = np.array(problem["matrix"], dtype=float)
        n = matrix.shape[0]
        
        if n <= 50:
            # For small-medium matrices, use LAPACK symmetric eigenvalue solver
            # with subset_by_value to only get eigenvalues near zero
            try:
                # Get eigenvalues in a range around zero
                # We use a generous range that should capture the 2 nearest to 0
                vals = scipy_eigvalsh(matrix, subset_by_index=[0, min(n-1, 3)])
                # Also get a few from the negative side
                all_candidates = vals.tolist()
                # Get a few largest (most negative) eigenvalues too
                if n > 4:
                    top_vals = scipy_eigvalsh(matrix, subset_by_index=[max(0, n-4), n-1])
                    all_candidates.extend(top_vals.tolist())
                all_candidates_sorted = sorted(all_candidates, key=abs)
                return all_candidates_sorted[:2]
            except Exception:
                eigenvalues = np.linalg.eigvalsh(matrix)
                eigenvalues_sorted = sorted(eigenvalues.tolist(), key=abs)
                return eigenvalues_sorted[:2]
        
        try:
            # For large matrices, use shift-invert ARPACK
            vals = eigsh(matrix, k=min(4, n-1), sigma=0, which='LM', 
                        return_eigenvectors=False, tol=1e-7, maxiter=1000)
            vals_sorted = sorted(vals.tolist(), key=abs)
            return vals_sorted[:2]
        except Exception:
            eigenvalues = np.linalg.eigvalsh(matrix)
            eigenvalues_sorted = sorted(eigenvalues.tolist(), key=abs)
            return eigenvalues_sorted[:2]