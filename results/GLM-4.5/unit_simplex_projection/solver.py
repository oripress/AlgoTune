import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the Euclidean projection onto the probability simplex problem.
        
        :param problem: A dictionary of the problem's parameters.
        :return: A dictionary with key "solution" containing the optimal projection.
        """
        y = np.asarray(problem["y"], dtype=np.float64)
        n = len(y)
        
        # Handle edge cases
        if n == 0:
            return {"solution": np.array([])}
        if n == 1:
            return {"solution": np.array([1.0])}
        
        # For large n, use partial sort to get top k elements
        if n > 15:
            # Estimate k based on the sum constraint
            k = min(n, max(3, int(n * 0.03)))
            # Get indices of top k elements
            top_k_indices = np.argpartition(y, -k)[-k:]
            # Sort only the top k elements
            top_k_values = y[top_k_indices]
            sorted_top_k = np.sort(top_k_values)[::-1]
            
            # Compute cumulative sum for top k
            cumsum_top_k = np.cumsum(sorted_top_k) - 1
            
            # Find rho in top k
            rho_top_k = np.where(sorted_top_k > cumsum_top_k / (np.arange(1, k + 1)))[0]
            if len(rho_top_k) > 0:
                rho = rho_top_k[-1]
                theta = cumsum_top_k[rho] / (rho + 1)
            else:
                # All elements are small, use uniform distribution
                theta = (np.sum(y) - 1.0) / n
        else:
            # For small n, use full sort
            sorted_y = np.sort(y)[::-1]
            cumsum_y = np.cumsum(sorted_y) - 1
            rho = np.where(sorted_y > cumsum_y / (np.arange(1, n + 1)))[0][-1]
            theta = cumsum_y[rho] / (rho + 1)
        
        x = np.maximum(y - theta, 0)
        
        return {"solution": x}