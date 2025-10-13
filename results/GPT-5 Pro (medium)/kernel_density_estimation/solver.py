from __future__ import annotations

from typing import Any, Dict, List

import math
import numpy as np

class Solver:
    """
    Fast Kernel Density Estimation solver with vectorized computations.

    Supports kernels: 'gaussian', 'tophat', 'epanechnikov' via optimized numpy.
    Falls back to sklearn.neighbors.KernelDensity for any other kernel to ensure correctness.
    """

    # Reasonable memory budget for intermediate pairwise blocks (in floats)
    # 4_000_000 floats ~ 32 MB
    _TARGET_ELEMS: int = 4_000_000

    def __init__(self) -> None:
        # Kernels we implement directly
        self._fast_kernels = {"gaussian", "tophat", "epanechnikov"}

    @staticmethod
    def _log_volume_unit_ball(d: int) -> float:
        # log V_d where V_d = pi^(d/2) / Gamma(d/2 + 1)
        return (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1.0)

    @staticmethod
    def _pairwise_block_params(m: int, n: int, target_elems: int) -> tuple[int, int]:
        """
        Choose block sizes (rows B, cols K) so that B*K <= target_elems and both are >= 1,
        while keeping B not too small to exploit BLAS performance.
        """
        if m <= 0 or n <= 0:
            return 0, 0
        # Start with a reasonable row block size
        B = min(m, 1024)
        # Compute column block size to fit memory budget
        K = max(1, min(n, target_elems // max(1, B)))
        if K * B > target_elems and K > 1:
            K = max(1, target_elems // max(1, B))
        # If K is tiny, try reducing B to allow larger K
        if K <= 4 and B > 64:
            B = max(64, target_elems // max(1, K))
            B = min(B, m)
            K = max(1, min(n, target_elems // max(1, B)))
        B = max(1, min(B, m))
        K = max(1, min(K, n))
        return B, K

    def _score_gaussian(self, X: np.ndarray, Q: np.ndarray, h: float) -> np.ndarray:
        n, d = X.shape
        m = Q.shape[0]
        h2 = h * h

        # Constants for Gaussian kernel
        const = -math.log(n) - d * math.log(h) - 0.5 * d * math.log(2.0 * math.pi)

        X_norm2 = np.sum(X * X, axis=1)  # (n,)
        B, K = self._pairwise_block_params(m, n, self._TARGET_ELEMS)

        out = np.empty(m, dtype=np.float64)

        # Process queries in blocks
        for i0 in range(0, m, B):
            i1 = min(i0 + B, m)
            Qb = Q[i0:i1]  # (b, d)
            Qb_norm2 = np.sum(Qb * Qb, axis=1)  # (b,)
            b = i1 - i0

            # Streaming log-sum-exp accumulators
            M = None  # row-wise max accumulator
            S = None  # row-wise sum of exp shifted by M

            for j0 in range(0, n, K):
                j1 = min(j0 + K, n)
                Xj = X[j0:j1]  # (k, d)
                Xj_norm2 = X_norm2[j0:j1]  # (k,)

                # Compute pairwise squared distances using dot trick, in-place reuse
                # Dist2 = ||q||^2 + ||x||^2 - 2 q.x
                Dot = Qb @ Xj.T  # (b, k)
                np.multiply(Dot, -2.0, out=Dot)  # Dot = -2 * Dot
                Dot += Qb_norm2[:, None]
                Dot += Xj_norm2[None, :]

                # A = -Dist2 / (2 h^2)
                A = Dot
                A *= -(0.5 / h2)

                # Row-wise max and sum-exp for this chunk
                Amax = np.max(A, axis=1)  # (b,)
                # Shift and exp in-place
                A -= Amax[:, None]
                np.exp(A, out=A)  # now A holds exp(A - Amax)
                chunk_sum = np.sum(A, axis=1)  # (b,)

                if M is None:
                    M = Amax
                    S = chunk_sum
                else:
                    # Combine with previous using log-sum-exp streaming
                    M_new = np.maximum(M, Amax)
                    # S = S*exp(M - M_new) + chunk_sum*exp(Amax - M_new)
                    S *= np.exp(M - M_new)
                    S += chunk_sum * np.exp(Amax - M_new)
                    M = M_new

            # lse = M + log(S)
            lse = M + np.log(S)
            out[i0:i1] = lse + const

        return out

    def _score_tophat(self, X: np.ndarray, Q: np.ndarray, h: float) -> np.ndarray:
        n, d = X.shape
        m = Q.shape[0]
        h2 = h * h

        # Constants: 1/(n h^d V_d)
        log_Vd = self._log_volume_unit_ball(d)
        const = -math.log(n) - d * math.log(h) - log_Vd

        X_norm2 = np.sum(X * X, axis=1)  # (n,)
        B, K = self._pairwise_block_params(m, n, self._TARGET_ELEMS)

        out = np.full(m, -np.inf, dtype=np.float64)

        for i0 in range(0, m, B):
            i1 = min(i0 + B, m)
            Qb = Q[i0:i1]
            Qb_norm2 = np.sum(Qb * Qb, axis=1)
            b = i1 - i0

            counts = np.zeros(b, dtype=np.int64)

            for j0 in range(0, n, K):
                j1 = min(j0 + K, n)
                Xj = X[j0:j1]
                Xj_norm2 = X_norm2[j0:j1]

                Dot = Qb @ Xj.T  # (b, k)
                np.multiply(Dot, -2.0, out=Dot)
                Dot += Qb_norm2[:, None]
                Dot += Xj_norm2[None, :]

                # Count points within radius h (<= h^2)
                counts += np.sum(Dot <= h2, axis=1)

            mask = counts > 0
            if np.any(mask):
                out[i0:i1][mask] = np.log(counts[mask].astype(np.float64)) + const

        return out

    def _score_epanechnikov(self, X: np.ndarray, Q: np.ndarray, h: float) -> np.ndarray:
        n, d = X.shape
        m = Q.shape[0]
        h2 = h * h

        # K(u) = c_d * (1 - ||u||^2) for ||u|| <= 1 else 0, with c_d = (d+2)/(2 V_d)
        log_Vd = self._log_volume_unit_ball(d)
        log_cd = math.log(d + 2.0) - math.log(2.0) - log_Vd
        const = -math.log(n) - d * math.log(h) + log_cd

        X_norm2 = np.sum(X * X, axis=1)  # (n,)
        B, K = self._pairwise_block_params(m, n, self._TARGET_ELEMS)

        out = np.full(m, -np.inf, dtype=np.float64)

        for i0 in range(0, m, B):
            i1 = min(i0 + B, m)
            Qb = Q[i0:i1]
            Qb_norm2 = np.sum(Qb * Qb, axis=1)
            b = i1 - i0

            sums = np.zeros(b, dtype=np.float64)

            for j0 in range(0, n, K):
                j1 = min(j0 + K, n)
                Xj = X[j0:j1]
                Xj_norm2 = X_norm2[j0:j1]

                Dot = Qb @ Xj.T  # (b, k)
                np.multiply(Dot, -2.0, out=Dot)
                Dot += Qb_norm2[:, None]
                Dot += Xj_norm2[None, :]

                # r2 = Dist2 / h^2
                Dot /= h2
                # weights = max(1 - r2, 0)
                np.subtract(1.0, Dot, out=Dot)  # Dot = 1 - r2
                np.clip(Dot, 0.0, None, out=Dot)  # zero out negative values
                sums += np.sum(Dot, axis=1)

            mask = sums > 0.0
            if np.any(mask):
                out[i0:i1][mask] = np.log(sums[mask]) + const

        return out

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        try:
            X = np.asarray(problem["data_points"], dtype=np.float64)
            Q = np.asarray(problem["query_points"], dtype=np.float64)
            kernel = str(problem["kernel"]).lower()
            bandwidth = float(problem["bandwidth"])

            # Basic shape validations
            if X.ndim != 2 or Q.ndim != 2:
                raise ValueError("Data points or query points are not 2D arrays.")
            if X.shape[0] == 0:
                raise ValueError("No data points provided.")
            if Q.shape[0] == 0:
                return {"log_density": []}
            if X.shape[1] != Q.shape[1]:
                raise ValueError("Data points and query points have different dimensions.")
            if not (bandwidth > 0.0) or not np.isfinite(bandwidth):
                raise ValueError("Bandwidth must be positive.")

            if kernel in self._fast_kernels:
                if kernel == "gaussian":
                    log_density = self._score_gaussian(X, Q, bandwidth)
                elif kernel == "tophat":
                    log_density = self._score_tophat(X, Q, bandwidth)
                else:  # epanechnikov
                    log_density = self._score_epanechnikov(X, Q, bandwidth)
            else:
                # Fallback to sklearn for any other kernel to ensure correctness
                from sklearn.neighbors import KernelDensity  # lazy import to reduce overhead

                kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
                kde.fit(X)
                log_density = kde.score_samples(Q)

            return {"log_density": log_density.tolist()}

        except KeyError as e:
            return {"error": f"Missing key: {e}"}
        except Exception as e:
            # Mirror the reference solver's broad exception handling
            return {"error": f"Computation error: {e}"}