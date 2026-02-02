from __future__ import annotations

from math import inf, lgamma, log, pi
from typing import Any

import numpy as np

try:
    from sklearn.neighbors import KernelDensity  # type: ignore
except Exception:  # pragma: no cover
    KernelDensity = None

class Solver:
    # Kernel names supported by sklearn.neighbors.KernelDensity
    available_kernels = {
        "gaussian",
        "tophat",
        "epanechnikov",
        "exponential",
        "linear",
        "cosine",
    }

    def __init__(self) -> None:
        # (kernel, bandwidth, dims, log_norm)
        self._norm_cache: tuple[str, float, int, float] | None = None

    @staticmethod
    def _log_unit_ball_volume(d: int) -> float:
        # V_d = pi^(d/2) / Gamma(d/2 + 1)
        return 0.5 * d * log(pi) - lgamma(0.5 * d + 1.0)

    @staticmethod
    def _log_unit_sphere_area(d: int) -> float:
        # S_{d-1} = 2 * pi^(d/2) / Gamma(d/2)
        return log(2.0) + 0.5 * d * log(pi) - lgamma(0.5 * d)

    def _log_kernel_norm(self, kernel: str, bandwidth: float, dims: int) -> float:
        """
        Return log of sklearn KDE normalization constant (without the 1/n factor).

        This matches sklearn's kernels for:
          gaussian, tophat, epanechnikov, exponential, linear.
        Cosine is handled via sklearn fallback in `solve`.
        """
        c = self._norm_cache
        if c is not None and c[0] == kernel and c[1] == bandwidth and c[2] == dims:
            return c[3]

        d = dims
        h = bandwidth
        log_h = log(h)

        if kernel == "gaussian":
            # (2*pi)^(-d/2) * h^(-d)
            log_norm = -(d / 2.0) * log(2.0 * pi) - d * log_h

        elif kernel == "tophat":
            # 1 / (V_d * h^d)
            log_norm = -self._log_unit_ball_volume(d) - d * log_h

        elif kernel == "epanechnikov":
            # (d+2) / (2 * V_d) * h^(-d)
            log_norm = log(d + 2.0) - log(2.0) - self._log_unit_ball_volume(d) - d * log_h

        elif kernel == "exponential":
            # 1 / (S_{d-1} * Gamma(d)) * h^(-d)
            log_norm = -self._log_unit_sphere_area(d) - lgamma(float(d)) - d * log_h

        elif kernel == "linear":
            # d(d+1) / S_{d-1} * h^(-d)
            log_norm = log(d * (d + 1.0)) - self._log_unit_sphere_area(d) - d * log_h

        else:
            log_norm = -inf

        log_norm_f = float(log_norm)
        self._norm_cache = (kernel, bandwidth, dims, log_norm_f)
        return log_norm_f

    @staticmethod
    def _logsumexp_rows_inplace(a: np.ndarray) -> np.ndarray:
        """
        Row-wise logsumexp over axis=1, modifying `a` in-place.

        Robust to rows where all entries are non-finite (e.g., all -inf), avoiding NaNs.
        """
        a_max = np.max(a, axis=1)
        finite = np.isfinite(a_max)
        out = np.empty_like(a_max)

        if np.any(finite):
            af = a[finite]
            mf = a_max[finite]
            af -= mf[:, None]
            np.exp(af, out=af)
            s = np.sum(af, axis=1)
            out[finite] = mf + np.log(s)

        out[~finite] = a_max[~finite]  # typically -inf
        return out

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        # Minimal-overhead wrapper around sklearn's KernelDensity (matches reference outputs).
        X = np.asarray(problem["data_points"], dtype=np.float64)
        Q = np.asarray(problem["query_points"], dtype=np.float64)

        m = int(Q.shape[0])
        if m == 0:
            return {"log_density": []}

        if KernelDensity is None:
            # Should not happen on the evaluation system.
            return {"log_density": np.full(m, -inf, dtype=np.float64)}

        kde = KernelDensity(
            kernel=problem["kernel"],
            bandwidth=float(problem["bandwidth"]),
        )
        kde.fit(X)
        return {"log_density": kde.score_samples(Q)}