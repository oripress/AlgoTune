import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
from scipy.special import logsumexp, gamma
from typing import Any
import numba

class Solver:
    def __init__(self):
        self.available_kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        try:
            X = np.array(problem["data_points"])
            X_q = np.array(problem["query_points"])
            kernel = problem["kernel"]
            bandwidth = problem["bandwidth"]

            if X.ndim != 2 or X_q.ndim != 2:
                raise ValueError("Data points or query points are not 2D arrays.")
            if X.shape[0] == 0:
                raise ValueError("No data points provided.")
            if X_q.shape[0] == 0:
                return {"log_density": []}
            if X.shape[1] != X_q.shape[1]:
                raise ValueError("Data points and query points have different dimensions.")

            bandwidth = float(bandwidth)
            if bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
            if kernel not in self.available_kernels:
                raise ValueError(f"Unknown kernel: {kernel}")

            N = X.shape[0]
            M = X_q.shape[0]
            d = X.shape[1]

            if kernel == 'cosine' or N * M > 100000:
                kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
                kde.fit(X)
                log_density = kde.score_samples(X_q)
                return {"log_density": log_density.tolist()}

            if kernel == 'gaussian':
                dists_sq = cdist(X_q, X, metric='sqeuclidean')
                log_norm_factor = d * np.log(bandwidth * np.sqrt(2 * np.pi))
                log_density = logsumexp(-0.5 * dists_sq / (bandwidth**2), axis=1) - log_norm_factor - np.log(N)
            else:
                dists = cdist(X_q, X, metric='euclidean')
                V_d = (np.pi ** (d / 2)) / gamma(d / 2 + 1) * (bandwidth ** d)
                
                if kernel == 'tophat':
                    K = (dists <= bandwidth).astype(float)
                    density = K.sum(axis=1) / (N * V_d)
                elif kernel == 'epanechnikov':
                    norm_factor = (d + 2) / (2 * V_d)
                    u = dists / bandwidth
                    K = np.maximum(0, 1 - u**2)
                    density = K.sum(axis=1) * norm_factor / N
                elif kernel == 'exponential':
                    norm_factor = 1 / (gamma(d + 1) * V_d)
                    u = dists / bandwidth
                    K = np.exp(-u)
                    density = K.sum(axis=1) * norm_factor / N
                else: # linear
                    norm_factor = (d + 1) / V_d
                    u = dists / bandwidth
                    K = np.maximum(0, 1 - u)
                    density = K.sum(axis=1) * norm_factor / N
                
                with np.errstate(divide='ignore'):
                    log_density = np.log(density)

            return {"log_density": log_density.tolist()}
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            print("ERROR:", err)
            return {"error": err}