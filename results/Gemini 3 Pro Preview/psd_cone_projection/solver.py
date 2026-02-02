from typing import Any
import numpy as np
try:
    import fast_reconstruct
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Solves a given positive semidefinite cone projection problem.

        Args:
            problem: A dictionary with problem parameter:
                - A: symmetric matrix.

        Returns:
            A dictionary containing the problem solution:
                - X: result of projecting A onto PSD cone.
        """
        A = np.array(problem["A"])
        # eigh returns eigenvalues in ascending order
        eigvals, eigvecs = np.linalg.eigh(A)
        
        # Filter positive eigenvalues
        k = np.searchsorted(eigvals, 1e-12)
        
        if k == len(eigvals):
            return {"X": np.zeros_like(A)}
            
        vals = eigvals[k:]
        vecs = eigvecs[:, k:]
        
        if HAS_CYTHON:
            # Ensure arrays are C-contiguous and float64
            if not vals.flags.c_contiguous:
                vals = np.ascontiguousarray(vals, dtype=np.float64)
            else:
                vals = vals.astype(np.float64, copy=False)
                
            if not vecs.flags.c_contiguous:
                vecs = np.ascontiguousarray(vecs, dtype=np.float64)
            else:
                vecs = vecs.astype(np.float64, copy=False)
                
            X = fast_reconstruct.reconstruct_cython(vecs, vals)
        else:
            X = (vecs * vals) @ vecs.T
        
        return {"X": X}