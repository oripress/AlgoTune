from typing import Any
import numpy as np
from sklearn.linear_model import Lasso

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        # Use sklearn’s optimized Cython coordinate-descent for Lasso
        clf = Lasso(alpha=0.1, fit_intercept=False, max_iter=1000, tol=1e-4)
        clf.fit(X, y)
        return clf.coef_.tolist()

def solve(problem: dict[str, Any], **kwargs) -> list[float]:
    return Solver().solve(problem, **kwargs)