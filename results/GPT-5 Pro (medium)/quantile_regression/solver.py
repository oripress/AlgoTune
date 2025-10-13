from typing import Any, Dict

import numpy as np
from sklearn.linear_model import QuantileRegressor

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Fit quantile regression with scikit-learn and return parameters +
        in-sample predictions.

        :param problem: dict with keys "X", "y", "quantile", "fit_intercept"
        :return: dict with 'coef', 'intercept', 'predictions'
        """
        X = np.array(problem["X"], dtype=float, copy=False)
        y = np.array(problem["y"], dtype=float, copy=False)

        model = QuantileRegressor(
            quantile=float(problem["quantile"]),
            alpha=0.0,  # no ℓ2 shrinkage
            fit_intercept=bool(problem["fit_intercept"]),
            solver="highs",
        )
        model.fit(X, y)

        coef = model.coef_.tolist()
        intercept = [float(model.intercept_)]
        predictions = model.predict(X).tolist()

        return {"coef": coef, "intercept": intercept, "predictions": predictions}