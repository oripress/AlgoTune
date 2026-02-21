import numpy as np
from scipy.interpolate._rbfinterp import _get_backend, _NAME_TO_MIN_DEGREE, _SCALE_INVARIANT
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        x_train = np.asarray(problem["x_train"], float)
        y_train = np.asarray(problem["y_train"], float).ravel()
        x_test = np.asarray(problem["x_test"], float)

        rbf_config = problem.get("rbf_config", {})
        kernel = rbf_config.get("kernel", "thin_plate_spline").lower()
        epsilon = rbf_config.get("epsilon")
        smoothing = rbf_config.get("smoothing", 0.0)

        if epsilon is None:
            if kernel in _SCALE_INVARIANT:
                epsilon = 1.0
            else:
                epsilon = 1.0
        else:
            epsilon = float(epsilon)

        degree = max(_NAME_TO_MIN_DEGREE.get(kernel, -1), 0)

        ny, ndim = x_train.shape
        d = y_train.reshape(ny, 1)
        
        if isinstance(smoothing, (int, float)):
            smoothing = np.full(ny, smoothing, dtype=float)

        backend = _get_backend(np)
        powers = backend._monomial_powers(ndim, degree, np)
        
        shift, scale, coeffs = backend._build_and_solve_system(
            x_train, d, smoothing, kernel, epsilon, powers, np
        )
        
        out = backend.compute_interpolation(
            x_test, x_train, kernel, epsilon, powers, shift, scale, coeffs, np
        )
        
        return {"y_pred": out.ravel().tolist()}