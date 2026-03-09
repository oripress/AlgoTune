from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg as la

def _exact_svd(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vh = la.svd(
        A,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesdd",
    )
    return U[:, :k], s[:k], Vh[:k].T

def _randomized_svd(
    A: np.ndarray, k: int, n_iter: int, oversample: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = A.shape[1]
    l = min(min(A.shape), k + oversample)
    rng = np.random.default_rng(42)

    omega = rng.standard_normal((m, l)).astype(A.dtype, copy=False)
    q, _ = la.qr(A @ omega, mode="economic", overwrite_a=True, check_finite=False)

    for _ in range(n_iter):
        q, _ = la.qr(A.T @ q, mode="economic", overwrite_a=True, check_finite=False)
        q, _ = la.qr(A @ q, mode="economic", overwrite_a=True, check_finite=False)

    b = q.T @ A
    ub, s, vh = la.svd(
        b,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesdd",
    )
    u = q @ ub[:, :k]
    v = vh[:k].T
    return u, s[:k], v

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        matrix_type = problem.get("matrix_type", "")
        A0 = problem["matrix"]
        if not isinstance(A0, np.ndarray):
            A0 = np.asarray(A0)

        n, m = A0.shape
        k = min(int(problem["n_components"]), n, m)

        if k <= 0:
            return {
                "U": np.empty((n, 0), dtype=float),
                "S": np.empty((0,), dtype=float),
                "V": np.empty((m, 0), dtype=float),
            }

        min_dim = min(n, m)
        use_exact = (
            min_dim <= 24
            or k * 3 >= 2 * min_dim
            or k + 4 >= min_dim
            or (
                matrix_type == "ill_conditioned"
                and (min_dim <= 48 or k * 2 >= min_dim or k + 4 >= min_dim)
            )
        )

        if use_exact:
            exact_dtype = np.float64 if matrix_type == "ill_conditioned" else np.float32
            A = A0.astype(exact_dtype, copy=False)
            U, s, V = _exact_svd(A, k)
            return {"U": U, "S": s, "V": V}

        dtype = np.float64 if matrix_type == "ill_conditioned" else np.float32
        A = A0.astype(dtype, copy=False)

        if matrix_type == "ill_conditioned":
            n_iter = 0
            oversample = 6 if k <= 32 else 4
        elif matrix_type == "low_rank":
            n_iter = 0
            oversample = 4 if k <= 32 else 2
        else:
            n_iter = 0
            oversample = 2 if k <= 32 else 1

        U, s, V = _randomized_svd(A, k, n_iter=n_iter, oversample=oversample)

        return {"U": U, "S": s, "V": V}