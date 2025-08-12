import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Euclidean projection of a vector onto the probability simplex.

        Parameters
        ----------
        problem : dict
            Dictionary with key "y" containing the input vector as a list of numbers.

        Returns
        -------
        dict
            Dictionary with key "solution" containing the projected vector as a list.
        """
        y = np.asarray(problem.get("y", []), dtype=float)
        if y.ndim != 1:
            y = y.ravel()
        n = y.size

        # Edge case: empty input
        if n == 0:
            return {"solution": []}

        # Sort in descending order
        sorted_y = np.sort(y)[::-1]

        # Compute cumulative sum of sorted_y minus 1
        cumsum = np.cumsum(sorted_y) - 1.0

        # Find rho: the last index where sorted_y > cumsum / (index+1)
        # Vectorized computation
        idx = np.arange(1, n + 1)
        condition = sorted_y > (cumsum / idx)
        rho = np.where(condition)[0][-1]

        # Compute theta
        theta = cumsum[rho] / (rho + 1)

        # Project onto simplex
        x = np.maximum(y - theta, 0.0)

        return {"solution": x.tolist()}