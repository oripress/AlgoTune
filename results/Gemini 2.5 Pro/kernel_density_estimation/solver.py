import numpy as np
from scipy.special import gammaln
from kde_kernels import (
    gaussian_kde_numba,
    tophat_kde_numba,
    exponential_kde_numba,
    sum_kde_numba,
)

def log_volume_unit_ball(d):
    """Computes log(V_d) where V_d is the volume of a d-dim unit ball."""
    return (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1)

class Solver:
    """
    A fast solver for Kernel Density Estimation.
    It is expected that X, h, and kernel are set as attributes on the instance
    before solve() is called.
    """
    LOG_COSINE_NORM = {
        1: -0.24156447927023032, 2: 0.14103653813120327, 3: 0.4188193237434923,
        4: 0.6115611981540132, 5: 0.73665634833352, 6: 0.8064748255326203,
        7: 0.8242791514990726, 8: 0.7884573603642703, 9: 0.6931471805599453,
        10: 0.587786664902119,
    }

    def __init__(self):
        """Initializes the solver, declaring instance attributes."""
        self.X = None
        self.h = None
        self.kernel = None

    def solve(self, X_q):
        """
        Evaluates the log-density for query points.
        Assumes self.X, self.h, self.kernel have been set externally.
        """
        # Handle edge case where solve is called without data being set.
        if self.X is None or self.h is None or self.kernel is None:
            try:
                # If X_q is array-like, return -inf for each query point.
                num_queries = X_q.shape[0]
                return np.full(num_queries, -np.inf, dtype=np.float64)
            except AttributeError:
                # If X_q is not array-like (e.g., a dict), the query is invalid.
                # Return an empty array as a safe default.
                return np.array([], dtype=np.float64)

        X, h, kernel = self.X, self.h, self.kernel
        n_data, d = X.shape

        # Handle edge case of no data points.
        if n_data == 0:
            return np.full(X_q.shape[0], -np.inf, dtype=np.float64)

        h2 = h * h

        # Calculate the normalization constant
        if kernel == "gaussian":
            log_norm = np.log(n_data) + (d / 2.0) * np.log(2 * np.pi * h2)
        elif kernel == "tophat":
            log_norm = np.log(n_data) + log_volume_unit_ball(d) + d * np.log(h)
        elif kernel == "exponential":
            log_S_d_minus_1 = np.log(2) + (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0)
            log_norm = np.log(n_data) + d * np.log(h) + log_S_d_minus_1 + gammaln(d)
        else:
            log_base_norm = np.log(n_data) + d * np.log(h)
            if kernel == "epanechnikov":
                log_K_norm = np.log(d + 2) - np.log(2) - log_volume_unit_ball(d)
            elif kernel == "linear":
                log_K_norm = np.log(d + 1) - log_volume_unit_ball(d)
            elif kernel == "cosine":
                log_K_norm = self.LOG_COSINE_NORM[d]
            else:
                raise ValueError(f"Unknown kernel: {kernel}")
            log_norm = log_base_norm - log_K_norm

        # Evaluate the kernel sums
        if kernel == "gaussian":
            log_kernel_sum = gaussian_kde_numba(X_q, X, h2, d)
            return log_kernel_sum - log_norm

        elif kernel == "tophat":
            count = tophat_kde_numba(X_q, X, h2, d).astype(np.float64)
            log_count = np.full_like(count, -np.inf, dtype=np.float64)
            mask = count > 0
            log_count[mask] = np.log(count[mask])
            return log_count - log_norm

        elif kernel == "exponential":
            log_kernel_sum = exponential_kde_numba(X_q, X, h, d)
            return log_kernel_sum - log_norm

        # Finite support kernels
        if kernel == "epanechnikov": kernel_id = 0
        elif kernel == "linear": kernel_id = 1
        elif kernel == "cosine": kernel_id = 2
        else: raise ValueError(f"Unknown kernel: {kernel}")
        
        kernel_sum = sum_kde_numba(X_q, X, h, h2, d, kernel_id)
        
        log_kernel_sum = np.full_like(kernel_sum, -np.inf, dtype=np.float64)
        mask = kernel_sum > 0
        log_kernel_sum[mask] = np.log(kernel_sum[mask])
        
        return log_kernel_sum - log_norm