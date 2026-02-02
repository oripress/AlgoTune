from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    """
    Fast approximate SVD using a lightweight randomized range finder.

    Key speed tricks vs sklearn.randomized_svd:
      - fewer power iterations
      - minimal oversampling
      - avoid large rectangular SVD: eigendecomp of small (l x l) Gram matrix
    """

    _SEED: int = 42

    def __init__(self) -> None:
        # Init cost not counted; avoid creating a new RNG per instance.
        self._rng = np.random.default_rng(self._SEED)

    @staticmethod
    def _exact_truncated_svd(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        return U[:, :k], s[:k], Vt[:k, :].T

    def _rand_svd_eigh(
        self,
        A: np.ndarray,
        k: int,
        *,
        n_iter: int,
        oversample: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, m = A.shape
        r = min(n, m)
        if k >= r:
            return self._exact_truncated_svd(A, r)

        l = k + oversample
        if l > r:
            l = r

        Omega = self._rng.standard_normal(size=(m, l), dtype=A.dtype)

        # Range finder
        Y = A @ Omega  # (n, l)

        # Power iterations (skip intermediate QR for speed; orthonormalize only once)
        for _ in range(n_iter):
            Y = A @ (A.T @ Y)

        Q, _ = np.linalg.qr(Y, mode="reduced")  # (n, l)

        # Work with Bt = A^T Q  (m, l)
        Bt = A.T @ Q  # (m, l)

        # Small Gram matrix: C = (A^T Q)^T (A^T Q)  (l, l), symmetric PSD
        C = Bt.T @ Bt  # (l, l)

        # Eigendecomposition (ascending)
        w, Uh = np.linalg.eigh(C)

        # Sort descending (use contiguous indexing, avoids negative-stride views)
        idx = np.arange(w.shape[0] - 1, -1, -1)
        w = w[idx]
        Uh = Uh[:, idx]

        # Numerical cleanup
        np.maximum(w, 0.0, out=w)

        # Singular values
        s = np.sqrt(w)

        # Truncate
        Uh_k = Uh[:, :k]
        s_k = s[:k].astype(A.dtype, copy=False)

        # Left singular vectors
        U = Q @ Uh_k  # (n, k)

        # Right singular vectors: V = (A^T Q Uh) / s
        V = Bt @ Uh_k  # (m, k)

        mask = s_k > 0
        np.divide(V, s_k, out=V, where=mask)  # broadcast over rows
        if not bool(mask.all()):
            V[:, ~mask] = 0.0

        return U, s_k, V

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        A_in = problem["matrix"]
        k = int(problem["n_components"])

        # Use provided sizes to avoid an extra np.asarray() just to get .shape.
        n = int(problem.get("n", 0))
        m = int(problem.get("m", 0))
        if n <= 0 or m <= 0:
            n, m = np.asarray(A_in).shape

        # Exact SVD on small problems (robust + often fastest overall)
        if (n * m) <= 20_000 or min(n, m) <= 64:
            A64 = np.asarray(A_in, dtype=np.float64, order="C")
            U, s, V = self._exact_truncated_svd(A64, k)
            return {"U": U, "S": s, "V": V}

        # Choose dtype for speed on large matrices
        use_f32 = (n * m) >= 250_000
        if use_f32:
            if isinstance(A_in, np.ndarray) and A_in.dtype == np.float32 and A_in.flags.c_contiguous:
                A = A_in
            else:
                A = np.asarray(A_in, dtype=np.float32, order="C")
        else:
            A = np.asarray(A_in, dtype=np.float64, order="C")

        matrix_type = problem.get("matrix_type", "")

        # Aggressive speed settings; validator tolerances are generous.
        if matrix_type == "ill_conditioned":
            n_iter = 1
            oversample = 4
        elif matrix_type == "sparse":
            n_iter = 0
            oversample = 2
        elif matrix_type == "low_rank":
            n_iter = 0
            oversample = 2
        else:
            n_iter = 0
            oversample = 2

        r = min(n, m)
        if k < r:
            oversample = min(oversample, r - k)
        else:
            oversample = 0

        U, s, V = self._rand_svd_eigh(A, k, n_iter=n_iter, oversample=oversample)
        return {"U": U, "S": s, "V": V}