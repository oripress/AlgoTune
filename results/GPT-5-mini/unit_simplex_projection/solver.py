from typing import Any, Dict
import numpy as np

class Solver:
    def solve(self, problem: Any, **kwargs) -> Dict[str, np.ndarray]:
        """
        Project a vector y onto the probability simplex:
            minimize_x 0.5 * ||x - y||^2
            subject to x >= 0, sum(x) = 1

        Input:
            problem: dict with key "y" (array-like) or an array-like directly.
        Output:
            dict with key "solution" -> numpy.ndarray of shape (n,)
        """
        # Extract y
        if isinstance(problem, dict):
            y = problem.get("y", None)
        else:
            y = problem

        if y is None:
            return {"solution": np.array([], dtype=np.float64)}

        # Convert to 1D numpy array of float64
        y = np.asarray(y, dtype=np.float64).ravel()
        n = y.size
        if n == 0:
            return {"solution": np.array([], dtype=np.float64)}

        # Replace non-finite entries defensively (avoid NaN/inf propagation)
        if not np.all(np.isfinite(y)):
            # nan -> 0, positive inf -> large positive, negative inf -> large negative
            y = np.nan_to_num(y, nan=0.0, posinf=1e100, neginf=-1e100)

        # Fast path: already a valid probability vector
        if np.all(y >= 0.0) and np.isclose(y.sum(), 1.0, atol=1e-12):
            return {"solution": y.copy()}

        # Sort y in descending order
        u = np.sort(y)[::-1]

        # Compute cumulative sums and threshold candidates
        css = np.cumsum(u) - 1.0
        idx = np.arange(1, n + 1)
        t = css / idx

        # Find rho: largest index where u > t
        mask = u > t
        if np.any(mask):
            rho = np.nonzero(mask)[0][-1]
            theta = t[rho]
        else:
            # Numerical fallback: evenly distribute offset
            theta = (y.sum() - 1.0) / float(n)

        # Project onto the simplex
        x = y - theta
        x = np.maximum(x, 0.0)

        # Ensure no non-finite values
        x[~np.isfinite(x)] = 0.0

        # Defensive fallback: if numerical issues lead to zero sum, return uniform
        s = x.sum()
        if s <= 0.0 or not np.isfinite(s):
            x = np.full(n, 1.0 / n, dtype=np.float64)

        return {"solution": x}