import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the Euclidean projection onto the probability simplex problem.
        
        :param problem: A dictionary with key "y" containing the input vector
        :return: A dictionary with key "solution" containing the projection
        """
        y = np.array(problem["y"])
        
        # Ensure y is a 1D array
        y = y.flatten()
        n = len(y)
        
        # Sort y in descending order
        sorted_y = np.sort(y)[::-1]
        
        # Compute cumulative sums
        cumsum_y = np.cumsum(sorted_y) - 1
        
        # Find the threshold index
        rho = np.where(sorted_y > cumsum_y / (np.arange(1, n + 1)))[0][-1]
        theta = cumsum_y[rho] / (rho + 1)
        
        # Project onto the simplex
        x = np.maximum(y - theta, 0)
        
        return {"solution": x}