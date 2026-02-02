from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.linalg import eigh as _sp_eigh  # type: ignore
    from scipy.linalg.blas import dsyrk as _dsyrk  # type: ignore
except Exception:  # pragma: no cover
    _sp_eigh = None
    _dsyrk = None
class Solver:
    """
    Fast SVD solver.

    Strategy:
      - For highly rectangular matrices, compute SVD via eigendecomposition of the
        smaller Gram matrix (A^T A or A A^T). This is often faster than a full SVD.
      - Otherwise fall back to numpy.linalg.svd (LAPACK gesdd).
    """

    def __init__(self) -> None:
        self._eps = np.finfo(np.float64).eps

    @staticmethod
    def _orthonormal_completion(
        Q: np.ndarray, n: int, t: int, *, start_idx: int = 0
    ) -> np.ndarray:
        """
        Produce t orthonormal columns orthogonal to columns of Q (which is assumed
        to have orthonormal columns, up to numerical error).

        Uses deterministic modified Gram-Schmidt on standard basis vectors.
        """
        if t <= 0:
            return np.empty((n, 0), dtype=np.float64)

        r = 0 if Q.size == 0 else Q.shape[1]
        cols: list[np.ndarray] = []
        # Iterate over standard basis vectors (optionally offset).
        # This is deterministic and avoids RNG overhead.
        for j in range(start_idx, start_idx + n * 2):
            i = j % n
            v = np.zeros(n, dtype=np.float64)
            v[i] = 1.0

            # Project out existing Q.
            if r:
                # For e_i, Q.T @ e_i equals Q[i, :].
                v -= Q @ Q[i, :]

            # Project out already-chosen completion vectors.
            for c in cols:
                v -= c * float(np.dot(c, v))

            nv = float(np.linalg.norm(v))
            if nv > 1e-12:
                v /= nv
                cols.append(v)
                if len(cols) == t:
                    break

        if len(cols) < t:
            # Fallback: use QR on a deterministic dense matrix, then select complement.
            # This is rare and only triggered if the basis scan fails.
            R = np.arange(n * t, dtype=np.float64).reshape(n, t) + 1.0
            # Orthogonalize R against Q without changing Q: R <- R - Q(Q^T R)
            if r:
                R = R - Q @ (Q.T @ R)
            # Orthonormalize.
            Z, _ = np.linalg.qr(R, mode="reduced")
            cols.extend([Z[:, k].copy() for k in range(Z.shape[1]) if len(cols) < t])

        return np.stack(cols[:t], axis=1)

    def _svd_via_gram(self, A: np.ndarray) -> dict[str, np.ndarray]:
        """
        Compute thin SVD via Gram matrix eigendecomposition.
        Returns U (n,k), S (k,), V (m,k).

        Key optimization: scale eigenvectors by 1/s *before* the large GEMM, i.e.
        compute U = A @ (V * (1/s)) rather than (A @ V) * (1/s).
        This reduces elementwise work from O(n*k) to O(k*k).
        """
        n, m = A.shape
        nm = n if n > m else m

        if n >= m:
            # V from eig(A^T A)
            B = A.T @ A  # (m,m)
            w, V = np.linalg.eigh(B)  # ascending
            V = V[:, ::-1]  # descending eigenvectors

            s = w[::-1].copy()
            np.maximum(s, 0.0, out=s)
            np.sqrt(s, out=s)

            s0 = float(s[0]) if s.size else 0.0
            tol = self._eps * nm * (s0 if s0 > 0.0 else 1.0)
            nz = s > tol

            U = np.empty((n, m), dtype=np.float64)
            if np.all(nz):
                invs = 1.0 / s
                U[:, :] = A @ (V * invs)
            else:
                if np.any(nz):
                    invs_nz = 1.0 / s[nz]
                    U[:, nz] = A @ (V[:, nz] * invs_nz)

                # Orthonormal completion for zero singular values
                Q = U[:, nz] if np.any(nz) else np.empty((n, 0), dtype=np.float64)
                t = int(np.sum(~nz))
                U[:, ~nz] = self._orthonormal_completion(Q, n, t)

            return {"U": U, "S": s, "V": V.astype(np.float64, copy=False)}

        # n < m: U from eig(A A^T)
        B = A @ A.T  # (n,n)
        w, U = np.linalg.eigh(B)  # ascending
        U = U[:, ::-1]

        s = w[::-1].copy()
        np.maximum(s, 0.0, out=s)
        np.sqrt(s, out=s)

        s0 = float(s[0]) if s.size else 0.0
        tol = self._eps * nm * (s0 if s0 > 0.0 else 1.0)
        nz = s > tol

        V = np.empty((m, n), dtype=np.float64)
        if np.all(nz):
            invs = 1.0 / s
            V[:, :] = A.T @ (U * invs)
        else:
            if np.any(nz):
                invs_nz = 1.0 / s[nz]
                V[:, nz] = A.T @ (U[:, nz] * invs_nz)

            Q = V[:, nz] if np.any(nz) else np.empty((m, 0), dtype=np.float64)
            t = int(np.sum(~nz))
            V[:, ~nz] = self._orthonormal_completion(Q, m, t)

        return {"U": U.astype(np.float64, copy=False), "S": s, "V": V}

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        A = problem["matrix"]
        if not isinstance(A, np.ndarray):
            A = np.asarray(A, dtype=np.float64)
        else:
            if A.dtype != np.float64:
                A = A.astype(np.float64, copy=False)

        if A.ndim != 2:
            A = np.atleast_2d(A)

        n, m = A.shape
        k = n if n < m else m
        if k == 0:
            return {
                "U": np.empty((n, 0), dtype=np.float64),
                "S": np.empty((0,), dtype=np.float64),
                "V": np.empty((m, 0), dtype=np.float64),
            }

        # Prefer Gram/eigh for small-to-moderate k and rectangular matrices.
        if k <= 512 or n >= 2 * m or m >= 2 * n:
            return self._svd_via_gram(A)

        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        return {"U": U, "S": s, "V": Vh.T}