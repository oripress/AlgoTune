import numpy as np
from typing import Any, List
from sklearn.linear_model import Lasso

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        """
        Solve the Lasso regression problem:
        (1/(2n)) * ||y - Xw||^2_2 + alpha * ||w||_1
        with alpha = 0.1 and no intercept.
        """
        try:
            X = np.asarray(problem["X"], dtype=np.float64)
            y = np.asarray(problem["y"], dtype=np.float64)
            # Ensure y is 1‑dimensional
            if y.ndim > 1:
                y = y.ravel()
            # Fit Lasso regression (no intercept) matching reference implementation
            model = Lasso(alpha=0.1, fit_intercept=False, max_iter=1000, tol=1e-4)
            model.fit(X, y)
            return model.coef_.tolist()
        except Exception:
            # Fallback: zero vector of appropriate length
            d = np.shape(problem["X"])[1] if "X" in problem else 0
            return [0.0] * d