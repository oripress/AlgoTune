import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the sparse PCA problem.

        :param problem: Dictionary with problem parameters
        :return: Dictionary with the sparse principal components

        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        A = np.array(problem["covariance"])
        n_components = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])

        n = A.shape[0]  # Dimension of the data

        # Decision variables
        X = cp.Variable((n, n_components))

        # Use eigendecomposition-based approach for sparse PCA
        # Minimize ||B - X||_F^2 + λ ||X||_1 where B contains principal components

        # Get the eigendecomposition of A - optimized approach
        # Use eigh for symmetric matrices which is faster than eig
        # Get the eigendecomposition of A - optimized approach
        # Use eigh for symmetric matrices which is faster than eig
        eigvals, eigvecs = np.linalg.eigh(A)
        
        # Keep only positive eigenvalues for PSD approximation
        pos_mask = eigvals > 0
        eigvals = eigvals[pos_mask]
        eigvecs = eigvecs[:, pos_mask]
        
        # Sort in descending order using argsort for better performance
        idx = np.argsort(-eigvals)  # Negative for descending order
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Use the top n_components eigenvectors scaled by sqrt(eigenvalues)
        k = min(len(eigvals), n_components)
        sqrt_eigvals = np.sqrt(eigvals[:k])
        B = eigvecs[:, :k] * sqrt_eigvals
        # Constraints: each component has unit norm - use more efficient formulation
        # Use cp.norm2 instead of cp.norm for better performance
        constraints = [cp.norm2(X[:, i]) <= 1 for i in range(n_components)]
        # Expand ||B - X||_F^2 = ||B||_F^2 - 2*trace(B^T X) + ||X||_F^2
        # Since ||B||_F^2 is constant, we can minimize: -2*trace(B^T X) + ||X||_F^2 + λ ||X||_1
        # This formulation can be more computationally efficient
        constant_term = cp.sum_squares(B)  # This term is constant and doesn't affect optimization
        objective = cp.Minimize(-2 * cp.sum(cp.multiply(B, X)) + cp.sum_squares(X) + sparsity_param * cp.sum(cp.abs(X)))


        # Solve the problem with optimized parameters
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False, max_iters=20, feastol=1e-3, reltol=1e-3, abstol=1e-3)

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or X.value is None:
                return {"components": [], "explained_variance": []}

            # Calculate explained variance for each component - vectorized approach
            components = X.value
            # Vectorized calculation of explained variance: diag(X^T A X)
            explained_variance = np.diag(components.T @ A @ components).tolist()

            return {"components": components.tolist(), "explained_variance": explained_variance}

        except cp.SolverError as e:
            return {"components": [], "explained_variance": []}
        except Exception as e:
            return {"components": [], "explained_variance": []}