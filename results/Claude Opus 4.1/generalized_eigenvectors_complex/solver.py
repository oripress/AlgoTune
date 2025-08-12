import numpy as np
import scipy.linalg as la

class Solver:
    def solve(self, problem):
        """
        Solve the generalized eigenvalue problem A·x = λ B·x
        
        :param problem: Tuple (A, B) where A and B are n x n real matrices
        :return: (eigenvalues, eigenvectors)
        """
        # Solve the generalized eigenvalue problem - use check_finite=False for speed
        vals, vecs = la.eig(problem[0], problem[1], check_finite=False)
        
        # Normalize eigenvectors (vectorized) - use out parameter for in-place operation
        np.divide(vecs, np.linalg.norm(vecs, axis=0), out=vecs)
        
        # Sort by descending real part, then descending imaginary part
        idx = np.lexsort((-vals.imag, -vals.real))
        
        # Return sorted results as lists
        return (vals[idx].tolist(), vecs[:, idx].T.tolist())