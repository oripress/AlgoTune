import numpy as np
from scipy import linalg
from typing import Any
import numba

@numba.jit(nopython=True, cache=True)
def check_stability(eigenvalues_real, eigenvalues_imag):
    """Check if all eigenvalues have magnitude < 1."""
    for i in range(len(eigenvalues_real)):
        mag_sq = eigenvalues_real[i]**2 + eigenvalues_imag[i]**2
        if mag_sq >= 1.0:
            return False
    return True

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the Lyapunov stability analysis problem.
        """
        A = problem["A"]
        if not isinstance(A, np.ndarray):
            A = np.array(A, dtype=np.float64)
        else:
            A = A.astype(np.float64)
        
        # Quick stability check via eigenvalues
        eigenvalues = np.linalg.eigvals(A)
        is_stable = check_stability(eigenvalues.real, eigenvalues.imag)
        
        if not is_stable:
            return {"is_stable": False, "P": None}
        
        # If stable, solve discrete Lyapunov equation
        n = A.shape[0]
        
        # For small matrices, use direct solver
        if n <= 10:
            # Solve A^T P A - P = -I using vectorization
            # This is equivalent to (I - A^T ⊗ A^T) vec(P) = vec(I)
            I = np.eye(n)
            AT = A.T
            
            # Kronecker product approach
            kron = np.kron(AT, AT)
            I_kron = np.eye(n*n)
            
            # Solve (I - kron) vec(P) = vec(I)
            vec_I = I.reshape(-1)
            try:
                vec_P = np.linalg.solve(I_kron - kron, vec_I)
                P = vec_P.reshape(n, n)
                
                # Make symmetric (handle numerical errors)
                P = (P + P.T) / 2
                
                return {"is_stable": True, "P": P.tolist()}
            except:
                pass
        
        # Fallback to scipy for larger matrices
        try:
            P = linalg.solve_discrete_lyapunov(A.T, np.eye(n))
            return {"is_stable": True, "P": P.tolist()}
        except:
            return {"is_stable": False, "P": None}