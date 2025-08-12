import numpy as np
from numpy.linalg import solve
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve, cholesky, LinAlgError
from typing import Any, Callable

class Solver:
    @staticmethod
    def _get_kernel_func(kernel: str, epsilon: np.float64) -> Callable[[np.ndarray], np.ndarray]:
        if kernel == "multiquadric":
            return lambda r: np.sqrt(1 + (epsilon * r)**2)
        if kernel == "inverse_multiquadric":
            return lambda r: 1.0 / np.sqrt(1 + (epsilon * r)**2)
        if kernel == "gaussian":
            return lambda r: np.exp(-(epsilon * r)**2)
        if kernel == "linear":
            return lambda r: r
        if kernel == "cubic":
            return lambda r: r**3
        if kernel == "quintic":
            return lambda r: r**5
        if kernel == "thin_plate_spline":
            return lambda r: np.where(r > 0, r**2 * np.log(r), 0.0)
        raise ValueError(f"Unknown kernel: {kernel}")

    @staticmethod
    def _construct_P(X: np.ndarray, degree: int, dtype: type = np.float64) -> np.ndarray:
        n_samples, n_dims = X.shape
        if degree == -1:
            return np.empty((n_samples, 0), dtype=dtype)
        
        if degree == 1:
            P = np.ones((n_samples, 1 + n_dims), dtype=dtype)
            P[:, 1:] = X
            return P

        if degree == 2:
            n_poly_terms = 1 + n_dims + n_dims * (n_dims + 1) // 2
            P = np.empty((n_samples, n_poly_terms), dtype=dtype)
            
            P[:, 0] = 1.0
            P[:, 1:1+n_dims] = X
            
            col_idx = 1 + n_dims
            for i in range(n_dims):
                for j in range(i, n_dims):
                    P[:, col_idx] = X[:, i] * X[:, j]
                    col_idx += 1
            return P
            
        raise ValueError(f"Unsupported degree: {degree}")

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        dtype = np.float64
        
        x_train = np.asarray(problem["x_train"], dtype=dtype)
        y_train_orig = np.asarray(problem["y_train"], dtype=dtype)
        x_test = np.asarray(problem["x_test"], dtype=dtype)
        
        y_train_was_1d = y_train_orig.ndim == 1
        y_train = y_train_orig[:, np.newaxis] if y_train_was_1d else y_train_orig
        
        n_samples, n_dims = x_train.shape
        n_outputs = y_train.shape[1]

        rbf_config = problem.get("rbf_config", {})
        kernel = rbf_config.get("kernel", "gaussian")
        epsilon = dtype(rbf_config.get("epsilon", 1.0))
        smoothing = dtype(rbf_config.get("smoothing", 0.0))

        if kernel in ['gaussian', 'inverse_multiquadric']:
            degree = -1
        elif kernel in ['multiquadric', 'linear', 'cubic', 'quintic']:
            degree = 1
        elif kernel == 'thin_plate_spline':
            degree = 2
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        kernel_func = self._get_kernel_func(kernel, epsilon)
        d_train = cdist(x_train, x_train, 'euclidean')
        A = kernel_func(d_train)
        A.flat[::A.shape[0] + 1] += smoothing

        if degree == -1:
            try:
                L = cholesky(A, lower=True)
                weights = cho_solve((L, True), y_train)
            except LinAlgError:
                weights = solve(A, y_train)
            
            d_test = cdist(x_test, x_train, 'euclidean')
            B = kernel_func(d_test)
            y_pred = B @ weights
        else:
            P = self._construct_P(x_train, degree, dtype)
            n_poly = P.shape[1]

            M = np.zeros((n_samples + n_poly, n_samples + n_poly), dtype=dtype)
            M[:n_samples, :n_samples] = A
            M[:n_samples, n_samples:] = P
            M[n_samples:, :n_samples] = P.T

            b = np.zeros((n_samples + n_poly, n_outputs), dtype=dtype)
            b[:n_samples] = y_train

            z = solve(M, b)
            weights = z[:n_samples]
            poly_coeffs = z[n_samples:]

            d_test = cdist(x_test, x_train, 'euclidean')
            B = kernel_func(d_test)
            P_test = self._construct_P(x_test, degree, dtype)
            y_pred = B @ weights + P_test @ poly_coeffs

        if y_train_was_1d:
            y_pred = y_pred.ravel()

        return {"y_pred": y_pred.tolist()}