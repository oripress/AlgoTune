from __future__ import annotations

from math import log, pi
from numbers import Real
from typing import Any

import numpy as np
from sklearn.neighbors import KDTree, KernelDensity

class Solver:
    def __init__(self) -> None:
        self.available_kernels = {
            "gaussian",
            "tophat",
            "epanechnikov",
            "exponential",
            "linear",
            "cosine",
        }

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        try:
            X = np.asarray(problem["data_points"], dtype=float)
            X_q = np.asarray(problem["query_points"], dtype=float)
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
            if not isinstance(bandwidth, Real):
                raise ValueError("Bandwidth must be positive.")
            bandwidth = float(bandwidth)
            if not np.isfinite(bandwidth) or bandwidth <= 0.0:
                raise ValueError("Bandwidth must be positive.")
            if kernel not in self.available_kernels:
                raise ValueError(f"Unknown kernel: {kernel}")

            _n_samples, _n_features = X.shape
            _n_queries = X_q.shape[0]
            log_density = self._tree_or_estimator_log_density(X, X_q, kernel, bandwidth)

            return {"log_density": log_density.tolist()}

        except KeyError as e:
            return {"error": f"Missing key: {e}"}
        except (ValueError, TypeError, np.linalg.LinAlgError) as e:
            return {"error": f"Computation error: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

    @staticmethod
    def _use_direct_gaussian(n_samples: int, n_queries: int, n_features: int) -> bool:
        pair_count = n_samples * n_queries
        work = pair_count * n_features
        return pair_count <= 25_000_000 and work <= 150_000_000

    @staticmethod
    def _tree_or_estimator_log_density(
        X: np.ndarray, X_q: np.ndarray, kernel: str, bandwidth: float
    ) -> np.ndarray:
        n_samples = X.shape[0]
        try:
            tree = KDTree(X)
            log_density = tree.kernel_density(
                X_q,
                h=bandwidth,
                kernel=kernel,
                atol=0.0,
                rtol=0.0,
                breadth_first=True,
                return_log=True,
            )
            return log_density - log(n_samples)
        except Exception:
            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
            kde.fit(X)
            return kde.score_samples(X_q)

    @staticmethod
    def _chunk_size(n_samples: int, target_pairs: int = 4_000_000) -> int:
        return max(1, min(target_pairs // max(n_samples, 1), 65536))

    @staticmethod
    def _logsumexp_rows(values: np.ndarray) -> np.ndarray:
        row_max = np.max(values, axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            sums = np.sum(np.exp(values - row_max[:, None]), axis=1)
        out = row_max + np.log(sums)
        bad = ~np.isfinite(row_max)
        if np.any(bad):
            out[bad] = -np.inf
        return out

    def _gaussian_direct(self, X: np.ndarray, X_q: np.ndarray, bandwidth: float) -> np.ndarray:
        n_samples, n_features = X.shape
        qn = X_q.shape[0]
        inv_h = 1.0 / bandwidth
        scale2 = inv_h * inv_h
        const_term = -log(n_samples) - n_features * log(bandwidth) - 0.5 * n_features * log(
            2.0 * pi
        )

        out = np.empty(qn, dtype=np.float64)
        x_sq = np.einsum("ij,ij->i", X, X)
        xt = X.T
        chunk = self._chunk_size(n_samples)

        for start in range(0, qn, chunk):
            end = min(qn, start + chunk)
            Q = X_q[start:end]
            q_sq = np.einsum("ij,ij->i", Q, Q)
            dots = Q @ xt
            dist2 = (q_sq[:, None] + x_sq[None, :] - 2.0 * dots) * scale2
            np.maximum(dist2, 0.0, out=dist2)
            out[start:end] = self._logsumexp_rows(-0.5 * dist2) + const_term

        return out