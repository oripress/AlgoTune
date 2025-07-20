from typing import Any
import numpy as np
from sklearn import linear_model

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Solves the Lasso regression problem.
        """
        X = np.array(problem["X"])
        y = np.array(problem["y"])
        
        clf = linear_model.Lasso(alpha=0.1, fit_intercept=False)
        clf.fit(X, y)
        
        return clf.coef_.tolist()