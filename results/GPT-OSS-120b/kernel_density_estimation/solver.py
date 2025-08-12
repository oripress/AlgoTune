import numpy as np
from typing import Any, Dict
from scipy.special import logsumexp
from math import pi, gamma, log

class Solver:
    """
    Fast Kernel Density Estimation solver.

    Implements KDE using pure NumPy and SciPy for numerical stability.
    Supports 'gaussian', 'tophat', and 'epanechnikov' kernels.
    Returns the logarithm of the estimated density at each query point.
    """

    # Kernel functions returning log‑kernel values
    @staticmethod
    def _gaussian_log_kernel(sq_norm: np.ndarray, d: int) -> np.ndarray:
        const = -0.5 * d * log(2 * pi)
        return const - 0.5 * sq_norm

    @staticmethod
    def _tophat_log_kernel(sq_norm: np.ndarray, d: int) -> np.ndarray:
        # Volume of the d‑dimensional unit ball
        vol = pi ** (d / 2) / gamma(d / 2 + 1)
        log_const = -log(vol)
        inside = sq_norm <= 1.0
        out = np.full_like(sq_norm, -np.inf)
        out[inside] = log_const
        return out

    @staticmethod
    def _epanechnikov_log_kernel(sq_norm: np.ndarray, d: int) -> np.ndarray:
        vol = pi ** (d / 2) / gamma(d / 2 + 1)
        c = (d + 2) / (2 * vol)
        inside = sq_norm <= 1.0
        out = np.full_like(sq_norm, -np.inf)
        vals = c * (1.0 - sq_norm[inside])
        vals = np.maximum(vals, 0.0)
        out[inside] = np.log(vals, where=vals > 0, out=np.full_like(vals, -np.inf))
        return out

    _kernel_funcs = {
        "gaussian": _gaussian_log_kernel,
        "tophat": _tophat_log_kernel,
        "epanechnikov": _epanechnikov_log_kernel,
    }

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Compute log‑density for each query point.

        Parameters
        ----------
        problem : dict
            Required keys:
                - "data_points": list of N points (each a list of length d)
                - "query_points": list of M points (each a list of length d)
                - "kernel": one of {"gaussian", "tophat", "epanechnikov"}
                - "bandwidth": positive float

        Returns
        -------
        dict
            {"log_density": [log p(x_q1), ..., log p(x_qM)]}
            or {"error": "..."} on failure.
        """
        try:
            # Accept both "data_points" and legacy "data" keys
            data_key = "data_points" if "data_points" in problem else "data"
            query_key = "query_points" if "query_points" in problem else "query"
            X = np.asarray(problem[data_key], dtype=np.float64)
            X_q = np.asarray(problem[query_key], dtype=np.float64)

            # Early exit for empty query set
            if X_q.shape[0] == 0:
                return {"log_density": []}

            if X.ndim != 2 or X_q.ndim != 2:
                raise ValueError("Data and query points must be 2‑dimensional arrays.")
            if X.shape[0] == 0:
                raise ValueError("No data points provided.")
            if X.shape[1] != X_q.shape[1]:
                raise ValueError("Dimensionality mismatch between data and query points.")
            # Extract bandwidth and kernel
            h = float(problem["bandwidth"])
            kernel = problem["kernel"]
            if h <= 0:
                raise ValueError("Bandwidth must be positive.")
            if kernel not in self._kernel_funcs:
                raise ValueError(f"Unsupported kernel: {kernel}")
            n, d = X.shape

            # Compute scaled squared distances: ||(x_q - x_i) / h||^2
            diff = (X_q[:, np.newaxis, :] - X[np.newaxis, :, :]) / h
            sq_norm = np.einsum('mnd,mnd->mn', diff, diff)  # shape (M, N)

            # Apply kernel in log‑space
            log_K = self._kernel_funcs[kernel](sq_norm, d)  # (M, N)

            # Normalisation constant: log(1/(n * h^d))
            log_norm_const = -log(n) - d * log(h)

            # Log‑density for each query point
            log_density = log_norm_const + logsumexp(log_K, axis=1)

            return {"log_density": log_density.tolist()}

        except KeyError as e:
            return {"error": f"Missing key: {e}"}
        except Exception as e:
            return {"error": f"Computation error: {e}"}