from typing import Any
import numpy as np
from sklearn import linear_model

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        """Solve Lasso regression using optimized reference implementation."""
        # Check if input is valid first
        if not problem or "X" not in problem or "y" not in problem:
            return []
        
        try:
            # use sklearn.linear_model.Lasso to solve the task
            clf = linear_model.Lasso(
                alpha=0.1,
                fit_intercept=False,
                warm_start=True,
                max_iter=1500,
                random_state=42
            )
            clf.fit(problem["X"], problem["y"])
            return clf.coef_.tolist()
        except Exception as e:
            # Get dimensions without converting to numpy again
            try:
                d = len(problem["X"][0]) if problem["X"] is not None and len(problem["X"]) > 0 else 0
            except:
                d = 0
            return [0.0] * d  # return trivial answer