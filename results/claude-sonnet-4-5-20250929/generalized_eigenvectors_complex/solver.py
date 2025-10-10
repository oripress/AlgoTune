import numpy as np
import scipy.linalg as la

class Solver:
    def solve(self, problem):
        """
        Optimized solver: use np.linalg.solve + np.linalg.eig when B is invertible.
        Falls back to generalized solver if B is singular.
        """
        A, B = problem
        
        try:
            # When B is invertible: solve B^-1 A x = λ x (faster)
            B_inv_A = np.linalg.solve(B, A)
            eigenvalues, eigenvectors = np.linalg.eig(B_inv_A)
        except np.linalg.LinAlgError:
            # B is singular, use generalized eigenvalue solver
            eigenvalues, eigenvectors = la.eig(
                A, B, check_finite=False, overwrite_a=True, overwrite_b=True
            )
        
        # Normalize and sort
        eigenvectors /= np.linalg.norm(eigenvectors, axis=0)
        sort_indices = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        
        return (
            eigenvalues[sort_indices].tolist(),
            eigenvectors[:, sort_indices].T.tolist()
        )