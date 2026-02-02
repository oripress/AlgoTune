import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the sparse PCA problem efficiently using analytical solution.
        """
        try:
            A = np.array(problem["covariance"])
            n_components = int(problem["n_components"])
            sparsity_param = float(problem["sparsity_param"])
            
            n = A.shape[0]
            
            if n_components <= 0:
                 return {"components": [], "explained_variance": []}

            # Heuristic for choosing solver
            # eigsh is efficient for small k relative to n
            # eigh is efficient for dense matrices when k is large or n is small
            # We use a threshold of 0.2 ratio and n > 200 to switch to eigsh
            if n > 200 and n_components < n * 0.2:
                # Use eigsh
                # k must be < n for eigsh
                k_eig = min(n_components, n - 1)
                eigvals, eigvecs = eigsh(A, k=k_eig, which='LA')
            else:
                # Use eigh
                if n_components >= n:
                    eigvals, eigvecs = eigh(A, check_finite=False)
                else:
                    eigvals, eigvecs = eigh(A, subset_by_index=(n - n_components, n - 1), check_finite=False)
            
            # Filter positive eigenvalues
            pos_indices = eigvals > 0
            eigvals = eigvals[pos_indices]
            eigvecs = eigvecs[:, pos_indices]
            
            # Sort descending
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:, ::-1]
            
            k = len(eigvals)
            
            # Construct B
            B = eigvecs * np.sqrt(eigvals)
            
            # Analytical solution
            threshold = sparsity_param / 2.0
            
            # Soft thresholding in-place
            # U = sign(B) * max(|B| - th, 0)
            #   = B - sign(B)*th  if |B| > th else 0
            
            abs_B = np.abs(B)
            mask_zero = abs_B <= threshold
            
            # We can modify B in place to save memory/allocation
            B[mask_zero] = 0.0
            
            mask_nonzero = ~mask_zero
            if np.any(mask_nonzero):
                # B[mask] = B[mask] - sign(B[mask]) * threshold
                # Note: sign(B[mask]) is just B[mask] / abs_B[mask]
                # So B[mask] - (B[mask]/abs_B[mask]) * threshold
                # = B[mask] * (1 - threshold/abs_B[mask])
                
                # This avoids computing sign explicitly
                B[mask_nonzero] *= (1.0 - threshold / abs_B[mask_nonzero])
            
            U = B
            
            # Projection onto L2 ball
            norms = np.linalg.norm(U, axis=0)
            scale = np.maximum(norms, 1.0)
            
            # In-place division
            U /= scale[np.newaxis, :]
            X_k = U
            
            # Prepare full X matrix
            if k < n_components:
                X = np.zeros((n, n_components))
                X[:, :k] = X_k
            else:
                X = X_k
                
            # Calculate explained variance
            # Vectorized: sum(X * (A @ X), axis=0)
            AX = A @ X
            explained_variance = np.sum(X * AX, axis=0)
            
            return {
                "components": X.tolist(),
                "explained_variance": explained_variance.tolist()
            }
            
        except Exception as e:
            return {"components": [], "explained_variance": []}
        except Exception as e:
            # In case of any error, return empty lists as per reference behavior
            return {"components": [], "explained_variance": []}