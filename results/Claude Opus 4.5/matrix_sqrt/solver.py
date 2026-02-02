import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem, **kwargs):
        A = np.asarray(problem["matrix"], dtype=complex)
        n = A.shape[0]
        
        # For very small matrices, scipy is fine
        if n <= 2:
            try:
                X, _ = scipy.linalg.sqrtm(A, disp=False)
                return {"sqrtm": {"X": X.tolist()}}
            except:
                return {"sqrtm": {"X": []}}
        
        # Check if Hermitian (allows faster algorithm)
        diff = A - A.T.conj()
        is_hermitian = np.abs(diff).max() < 1e-10 * (np.abs(A).max() + 1e-12)
        
        if is_hermitian:
            try:
                eigenvalues, Q = np.linalg.eigh(A)
                
                # For positive semi-definite
                if eigenvalues.min() >= -1e-10:
                    eigenvalues = np.maximum(eigenvalues, 0)
                    sqrt_eig = np.sqrt(eigenvalues)
                    X = (Q * sqrt_eig) @ Q.conj().T
                    return {"sqrtm": {"X": X.tolist()}}
            except:
                pass
        
        # Try general eigendecomposition
        try:
            eigenvalues, V = np.linalg.eig(A)
            
            # Principal square root of eigenvalues
            sqrt_eig = np.sqrt(eigenvalues)
            # Ensure principal branch (non-negative real part)
            sqrt_eig = np.where(sqrt_eig.real < 0, -sqrt_eig, sqrt_eig)
            
            # X = V @ diag(sqrt_eig) @ V^-1
            V_D = V * sqrt_eig
            X = np.linalg.solve(V.T, V_D.T).T
            
            if np.all(np.isfinite(X)):
                return {"sqrtm": {"X": X.tolist()}}
        except:
            pass
        
        # Fall back to scipy
        try:
            X, _ = scipy.linalg.sqrtm(A, disp=False)
        except:
            return {"sqrtm": {"X": []}}
        return {"sqrtm": {"X": X.tolist()}}