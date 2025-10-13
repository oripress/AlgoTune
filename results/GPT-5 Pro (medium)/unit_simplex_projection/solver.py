from typing import Any, Dict
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, np.ndarray]:
        """
        Euclidean projection of a vector y onto the probability simplex:
            minimize_x 0.5 * ||x - y||^2
            subject to sum(x) = 1, x >= 0

        Fast active-set (Michelot) algorithm with a safe fallback to the
        O(n log n) sorting-based method. The active-set method is typically
        O(n) in practice.
        """
        y = np.asarray(problem.get("y", []), dtype=float).ravel()
        n = y.size

        if n == 0:
            return {"solution": y}

        # Quick path: already on the simplex
        s = float(y.sum())
        if s > 0.0 and abs(s - 1.0) <= 1e-12 and np.all(y >= 0):
            return {"solution": y.copy()}

        # Active-set iterations
        m = n
        max_iter = 100  # safety bound; typically converges in < 10 iterations
        for _ in range(max_iter):
            theta = (s - 1.0) / m
            mask = y > theta
            m_new = int(mask.sum())
            if m_new == m:
                # Converged
                x = y - theta
                # In-place clipping
                np.clip(x, 0.0, None, out=x)
                return {"solution": x}
            if m_new == 0:
                break  # fall back (should not happen in valid cases)
            s = float(y[mask].sum())
            m = m_new

        # Fallback: sorting-based algorithm (O(n log n))
        u = np.sort(y)[::-1]
        css = np.cumsum(u) - 1.0
        j = np.arange(1, n + 1, dtype=float)
        rho = np.nonzero(u > (css / j))[0][-1]
        theta = css[rho] / (rho + 1.0)
        x = y - theta
        np.clip(x, 0.0, None, out=x)
        return {"solution": x}