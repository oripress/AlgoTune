import numpy as np
from typing import Any, Dict, List, Union

class Solver:
    def __init__(self):
        # Pre-compute constants during initialization
        self.max_iter = 1000
        self.tol = 1e-6
        
    def solve(self, problem: Dict[str, Union[List[List[float]], List[int], float]]) -> Any:
        """Solve logistic regression with group lasso penalty using proximal gradient descent."""
        # Extract problem data
        X = np.array(problem["X"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        gl = np.array(problem["gl"])
        lba = problem["lba"]
        
        # Get dimensions
        n, p_plus_1 = X.shape
        p = p_plus_1 - 1  # number of features (excluding intercept)
        
        # Process groups
        unique_groups, inverse_indices, group_sizes = np.unique(gl, return_inverse=True, return_counts=True)
        num_groups = len(unique_groups)
        
        # Create group membership matrix
        group_idx = np.zeros((p, num_groups), dtype=bool)
        group_idx[np.arange(p), inverse_indices] = 1
        
        # Group weights (sqrt of group size)
        group_weights = np.sqrt(group_sizes)
        
        # Initialize parameters
        beta0 = 0.0
        beta = np.zeros(p)
        
        # Separate intercept and features
        X_intercept = X[:, 0:1]
        X_features = X[:, 1:]
        
        # Lipschitz constant estimation
        L = 0.25 * np.linalg.norm(X_features, ord=2) ** 2 / n + 1e-8
        step_size = 1.0 / L
        
        # Proximal gradient descent
        for iter_num in range(self.max_iter):
            # Compute linear predictor
            linear_pred = beta0 + X_features @ beta
            
            # Compute probabilities
            probs = 1.0 / (1.0 + np.exp(-linear_pred))
            
            # Compute gradients
            residual = probs - y
            grad_beta0 = np.mean(residual)
            grad_beta = X_features.T @ residual / n
            
            # Gradient step
            beta0_new = beta0 - step_size * grad_beta0
            beta_temp = beta - step_size * grad_beta
            
            # Proximal step (group soft-thresholding)
            beta_new = np.zeros_like(beta)
            for j in range(num_groups):
                group_mask = group_idx[:, j]
                beta_group = beta_temp[group_mask]
                group_norm = np.linalg.norm(beta_group)
                
                threshold = step_size * lba * group_weights[j]
                if group_norm > threshold:
                    beta_new[group_mask] = (1 - threshold / group_norm) * beta_group
            
            # Check convergence
            if np.linalg.norm(beta_new - beta) + abs(beta0_new - beta0) < self.tol:
                break
                
            beta = beta_new
            beta0 = beta0_new
        
        # Compute optimal value
        linear_pred = beta0 + X_features @ beta
        log_loss = -np.sum(y * linear_pred) + np.sum(np.log(1 + np.exp(linear_pred)))
        
        # Group lasso penalty
        penalty = 0.0
        for j in range(num_groups):
            group_mask = group_idx[:, j]
            penalty += group_weights[j] * np.linalg.norm(beta[group_mask])
        penalty *= lba
        
        optimal_value = log_loss + penalty
        
        return {
            "beta0": float(beta0),
            "beta": beta.tolist(),
            "optimal_value": float(optimal_value)
        }