import numpy as np
from scipy.linalg import solve_discrete_lyapunov

class Solver:
    def solve(self, problem, **kwargs):
        A = np.array(problem["A"], dtype=np.float64)
        n = A.shape[0]
        
        # Check stability by looking at eigenvalues
        eigenvalues = np.linalg.eigvals(A)
        if np.max(np.abs(eigenvalues)) >= 1:
            return {"is_stable": False, "P": None}
        
        # For small matrices, use Kronecker product approach (direct solve)
        if n <= 6:
            At = A.T
            # Solve (A^T ⊗ A^T - I) vec(P) = -vec(Q) where Q = I
            AtAt = np.kron(At, At)
            coef = AtAt - np.eye(n * n)
            vec_Q = np.eye(n).ravel()
            vec_P = np.linalg.solve(coef, -vec_Q)
            P = vec_P.reshape((n, n))
        else:
            # Use scipy for larger matrices
            P = solve_discrete_lyapunov(A.T, np.eye(n))
        
        # Make P symmetric (for numerical stability)
        P = (P + P.T) / 2
        
        return {"is_stable": True, "P": P.tolist()}