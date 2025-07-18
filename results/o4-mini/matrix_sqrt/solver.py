import numpy as np
_complex = np.complex128
_eye2 = np.eye(2, dtype=_complex)
from numpy.linalg import eig, eigh, solve

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the principal matrix square root with optimizations:
         - direct formula for 1x1 and 2x2 matrices
         - Hermitian case via numpy.linalg.eigh
         - general case via numpy.linalg.eig + solve
        """
        A_list = problem.get("matrix")
        if A_list is None:
            return {"sqrtm": {"X": []}}

        # Convert to complex numpy array
        A = np.array(A_list, dtype=_complex)
        # Validate square matrix
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return {"sqrtm": {"X": []}}

        n = A.shape[0]
        # Trivial 1x1
        if n == 1:
            x = np.sqrt(A[0, 0])
            X = np.array([[x]], dtype=_complex)
        # Analytic 2x2
        elif n == 2:
            a, b = A[0, 0], A[0, 1]
            c, d = A[1, 0], A[1, 1]
            T = a + d
            gamma = np.sqrt(a * d - b * c)
            # principal branch sqrt(T + 2*gamma)
            denom = np.sqrt(T + 2 * gamma)
            X = (A + gamma * _eye2) / denom
        else:
            # Hermitian case
            if np.allclose(A, A.conj().T, rtol=1e-8, atol=1e-10):
                D, V = eigh(A)
                sqrtD = np.sqrt(D)
                # V * sqrtD scales columns
                X = (V * sqrtD) @ V.conj().T
            else:
                # General diagonalizable case
                D, V = eig(A)
                sqrtD = np.sqrt(D)
                # V * sqrtD scales columns to form V diag(sqrtD)
                M = V * sqrtD
                # Solve V^T X^T = M^T => X = M @ V^{-1}
                X = solve(V.T, M.T).T

        return {"sqrtm": {"X": X.tolist()}}