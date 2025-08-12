import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Compute approximate SVD using optimized randomized methods."""
        # Extract problem parameters
        A = np.array(problem["matrix"], dtype=np.float32)  # Use float32 for speed
        n_components = problem["n_components"]
        
        # Use zero power iterations for maximum speed
        n_iter = 0  
        
        # Use a simplified and faster randomized SVD approach
        n, m = A.shape
        k = min(n_components, min(n, m))  # Ensure we don't ask for more components than possible
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Use a more efficient approach with fewer matrix multiplications
        if n > m:
            # For tall matrices, work with A directly
            Omega = np.random.normal(0, 1, (m, k)).astype(np.float32)  # Exact size for speed
            Y = A @ Omega
            
            # No power iteration for maximum speed
            Q, _ = np.linalg.qr(Y, mode='reduced')  # Use numpy's faster QR
            B = Q.T @ A
            
            # SVD on smaller matrix
            Ub, s, Vt = np.linalg.svd(B, full_matrices=False, compute_uv=True)
            
            # Extract components
            U = Ub[:, :k]
            sk = s[:k]
            Vk = Vt[:k, :].T
            # Final projection
            U = Q @ U
        else:
            # For wide or square matrices, work with A.T
            Omega = np.random.normal(0, 1, (n, k)).astype(np.float32)
            Y = A.T @ Omega
            
            # No power iteration for maximum speed
            Q, _ = np.linalg.qr(Y, mode='reduced')
            B = A @ Q
            
            # SVD on smaller matrix
            U_tilde, s, Vt = np.linalg.svd(B, full_matrices=False, compute_uv=True)
            
            # Extract components
            U = U_tilde[:, :k]
            sk = s[:k]
            V = Q @ Vt.T
            Vk = V[:, :k]
            
        return {"U": U.astype(np.float64), "S": sk.astype(np.float64), "V": Vk.astype(np.float64)}  # Convert back to float64