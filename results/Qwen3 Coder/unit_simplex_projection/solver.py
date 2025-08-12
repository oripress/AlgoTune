import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Solve the Euclidean projection onto the probability simplex problem.
        
        :param problem: A dictionary with key "y" containing the vector to project
        :return: A dictionary with key "solution" containing the projected vector
        """
        y = np.asarray(problem["y"], dtype=np.float64)
        
        # Ensure y is 1D
        if y.ndim != 1:
            y = y.flatten()
        
        n = len(y)
        
        # Sort y in descending order
        sorted_y = np.sort(y)[::-1]
        
        # Compute cumulative sum and find rho
        cumsum_y = np.cumsum(sorted_y) - 1
        valid = sorted_y > cumsum_y / np.arange(1, n + 1)
        rho_indices = np.where(valid)[0]
        rho = rho_indices[-1] if len(rho_indices) > 0 else 0
        theta = cumsum_y[rho] / (rho + 1)
            
        # Project onto the simplex
        x = np.maximum(y - theta, 0)
        
        return {"solution": x}
        
        return {"solution": x}