from typing import Any
import numpy as np
import math

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Fast vectorized Kernel Density Estimation (KDE) solver.
        Supports 'gaussian', 'tophat', and 'epanechnikov' kernels.
        Returns log-density for each query point.
        """
        try:
            # Load and validate data
            X = np.ascontiguousarray(problem["data_points"], dtype=np.float64)
            Q = np.ascontiguousarray(problem["query_points"], dtype=np.float64)
            # Support flat list-of-floats as 2D input for 1D dims
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if Q.ndim == 1:
                Q = Q.reshape(-1, 1)
            kernel = problem["kernel"].lower()
            bandwidth = problem["bandwidth"]
            if not isinstance(bandwidth, (float, int)) or bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
            h = float(bandwidth)
            inv_h = 1.0 / h

            # Validate dimensions
            if X.ndim != 2 or Q.ndim != 2:
                raise ValueError("Data points or query points are not 2D arrays.")
            n, d = X.shape
            m, dq = Q.shape
            if n == 0:
                raise ValueError("No data points provided.")
            if dq != d:
                raise ValueError("Data points and query points have different dimensions.")
            if m == 0:
                return {"log_density": []}

            # Fallback for unsupported kernels
            if kernel not in ("gaussian", "tophat", "epanechnikov"):
                from sklearn.neighbors import KernelDensity
                kde = KernelDensity(kernel=kernel, bandwidth=h).fit(X)
                logd = kde.score_samples(Q)
                return {"log_density": logd.tolist()}

            # Scale data by bandwidth
            Xs = X * inv_h
            Qs = Q * inv_h

            # Precompute for fast distance calculations
            Xs_T = np.ascontiguousarray(Xs.T)  # shape (d, n)
            Xs_sq = np.sum(Xs * Xs, axis=1)    # shape (n,)
            Qs_sq = np.sum(Qs * Qs, axis=1)    # shape (m,)

            # Common constants
            log_n = math.log(n)
            log_h = math.log(h)
            log_density = np.empty(m, dtype=np.float64)

            # Block size to limit memory footprint
            MAX_ELEMS = 10_000_000
            block_q = max(1, min(m, MAX_ELEMS // n))

            # Compute log-density by kernel
            if kernel == "gaussian":
                # constant: -0.5*d*log(2*pi) - d*log_h - log(n)
                log_const = -0.5 * d * math.log(2 * math.pi) - d * log_h - log_n
                for i in range(0, m, block_q):
                    j = min(i + block_q, m)
                    # pairwise squared distances
                    block_dot = Qs[i:j] @ Xs_T
                    S_h2 = Qs_sq[i:j, None] + Xs_sq[None, :] - 2.0 * block_dot
                    # log-sum-exp for gaussian
                    S = -0.5 * S_h2
                    S_max = np.max(S, axis=1)
                    ex = np.exp(S - S_max[:, None])
                    sum_ex = np.sum(ex, axis=1)
                    lse = S_max + np.log(sum_ex)
                    log_density[i:j] = lse + log_const

            elif kernel == "tophat":
                # V_d = pi^{d/2} / Gamma(d/2 + 1)
                log_vd = 0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d + 1.0)
                log_const = -log_n - d * log_h - log_vd
                for i in range(0, m, block_q):
                    j = min(i + block_q, m)
                    block_dot = Qs[i:j] @ Xs_T
                    S_h2 = Qs_sq[i:j, None] + Xs_sq[None, :] - 2.0 * block_dot
                    counts = np.sum(S_h2 <= 1.0, axis=1)
                    # log(counts) + const, with -inf for zeros
                    with np.errstate(divide='ignore'):
                        log_density[i:j] = np.where(
                            counts > 0,
                            np.log(counts) + log_const,
                            -np.inf
                        )

            elif kernel == "epanechnikov":
                # V_d = pi^{d/2} / Gamma(d/2 + 1)
                log_vd = 0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d + 1.0)
                # C = (d+2)/(2*V_d)
                log_C = math.log(d + 2.0) - math.log(2.0) - log_vd
                log_const = log_C - log_n - d * log_h
                for i in range(0, m, block_q):
                    j = min(i + block_q, m)
                    block_dot = Qs[i:j] @ Xs_T
                    S_h2 = Qs_sq[i:j, None] + Xs_sq[None, :] - 2.0 * block_dot
                    weight = (1.0 - S_h2) * (S_h2 <= 1.0)
                    M = np.sum(weight, axis=1)
                    with np.errstate(divide='ignore'):
                        log_density[i:j] = np.where(
                            M > 0.0,
                            np.log(M) + log_const,
                            -np.inf
                        )

            else:
                raise ValueError(f"Unknown kernel: {kernel}")

            return {"log_density": log_density.tolist()}

        except KeyError as e:
            return {"error": f"Missing key: {e}"}
        except (ValueError, TypeError) as e:
            return {"error": f"Computation error: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}