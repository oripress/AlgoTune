import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[complex]]]]:
        """
        Optimized matrix square root solver using eigendecomposition.
        
        Computes the principal matrix square root X such that X @ X = A.
        """
        A = problem["matrix"]
        
        # Convert to numpy array if needed
        if not isinstance(A, np.ndarray):
            A = np.array(A, dtype=complex)
        
        try:
            # Compute eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eig(A)
            
            # Take principal square root of eigenvalues
            sqrt_eigenvalues = np.sqrt(eigenvalues)
            
            # Reconstruct matrix: X = V * sqrt(D) * V^{-1}
            # Use solve instead of computing inverse (faster)
            # X = (V * sqrt(D)) @ V^{-1}
            # Solve V.T @ X.T = (V * sqrt(D)).T for X.T
            temp = eigenvectors * sqrt_eigenvalues
            X = np.linalg.solve(eigenvectors.T, temp.T).T
            
            return {"sqrtm": {"X": X.tolist()}}
        except Exception:
            return {"sqrtm": {"X": []}}