import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        # Unpack input matrices
        A_list, B_list = problem
        # Convert to NumPy arrays
        A = np.array(A_list, dtype=float)
        B = np.array(B_list, dtype=float)
        # Solve the standard eigenproblem for M = B^{-1} A
        M = np.linalg.solve(B, A)
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eig(M)
        # Sort by descending real part, then descending imaginary part
        idx = np.lexsort((-eigvals.imag, -eigvals.real))
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Return eigenvalues and eigenvectors (rows) as Python lists
        return eigvals.tolist(), eigvecs.T.tolist()