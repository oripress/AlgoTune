import random
from typing import Any
import numpy as np
from sklearn.linear_model import QuantileRegressor

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Fit quantile regression with scikit-learn and return parameters +
        in-sample predictions.

        :param problem: dict returned by generate_problem
        :return: dict with 'coef', 'intercept', 'predictions'
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        X = np.array(problem["X"], dtype=float)
        y = np.array(problem["y"], dtype=float)

        model = QuantileRegressor(
            quantile=problem["quantile"],
            alpha=0.0,  # no ℓ₂ shrinkage
            fit_intercept=problem["fit_intercept"],
            solver="highs-ds",  # different solver that might be faster
        )
        model.fit(X, y)

        coef = model.coef_.tolist()
        intercept = [float(model.intercept_)]  # keep same shape (1,)
        predictions = model.predict(X).tolist()

        return {"coef": coef, "intercept": intercept, "predictions": predictions}