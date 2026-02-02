import numpy as np
import scipy.linalg as la

class Solver:
    def solve(self, problem, **kwargs) -> list:
        A, B = problem
        n = A.shape[0]
        
        # Try to convert to standard eigenvalue problem: B^(-1) A x = λ x
        # This can be faster than the generalized eigensolver
        try:
            # Use LU decomposition to solve B^(-1) * A efficiently
            lu, piv = la.lu_factor(B, check_finite=False)
            # Compute B^(-1) A by solving B X = A for X
            C = la.lu_solve((lu, piv), A, check_finite=False)
            
            # Solve standard eigenvalue problem
            eigenvalues = la.eigvals(C, check_finite=False, overwrite_a=True)
        except:
            # Fallback to generalized eigensolver if B is singular
            scale_B = np.sqrt(np.linalg.norm(B))
            B_scaled = B / scale_B
            A_scaled = A / scale_B
            eigenvalues = la.eigvals(A_scaled, B_scaled, check_finite=False)
        
        # Sort eigenvalues: descending by real part, then by imaginary part
        indices = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        solution = eigenvalues[indices].tolist()
        
        return solution