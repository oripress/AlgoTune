from typing import Any, Dict
import numpy as np

# Try to use scikit-learn's optimized randomized_svd when available
try:
    from sklearn.utils.extmath import randomized_svd  # type: ignore
    _HAVE_SKLEARN = True
except Exception:
    randomized_svd = None
    _HAVE_SKLEARN = False

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Fast approximate truncated SVD:
         - For small problems or near-full rank use exact np.linalg.svd.
         - Prefer sklearn.randomized_svd with very few power iters when available.
         - Otherwise use a lightweight NumPy randomized SVD fallback (float32 for big matmuls).
        """
        if "matrix" not in problem:
            raise ValueError("problem must contain key 'matrix'")

        A = np.asarray(problem["matrix"])
        if A.ndim != 2:
            A = A.reshape((A.shape[0], -1))
        n, m = A.shape
        min_dim = min(n, m)

        k = int(problem.get("n_components", min_dim))
        k = max(0, min(k, min_dim))
        if k == 0:
            return {
                "U": np.zeros((n, 0), dtype=float),
                "S": np.zeros((0,), dtype=float),
                "V": np.zeros((m, 0), dtype=float),
            }

        matrix_type = problem.get("matrix_type", None)

        # Minimal power iterations for speed; ill-conditioned matrices get slightly more.
        n_iter = 2 if matrix_type == "ill_conditioned" else 0

        # Heuristic: exact SVD is often fastest & most reliable for small problems
        if min_dim <= 250 or k >= min_dim or (k / max(1, min_dim)) > 0.5:
            U_full, s_full, Vt_full = np.linalg.svd(np.asarray(A, dtype=float, order="C"), full_matrices=False)
            return {"U": U_full[:, :k], "S": s_full[:k].copy(), "V": Vt_full[:k, :].T}

        # Prefer sklearn's implementation when available (fast C path)
        if _HAVE_SKLEARN:
            try:
                U_skl, s_skl, Vt_skl = randomized_svd(A, n_components=k, n_iter=n_iter, random_state=42)
                return {
                    "U": np.asarray(U_skl[:, :k], dtype=float),
                    "S": np.asarray(s_skl[:k].copy(), dtype=float),
                    "V": np.asarray(Vt_skl[:k, :].T, dtype=float),
                }
            except Exception:
                # Fall back to numpy randomized path on any failure
                pass

        # NumPy randomized SVD fallback
        oversample = 7
        l = min(min_dim, k + oversample)

        # Use float32 for very large matmuls to speed up computation
        use_float32 = (n * m) >= 10_000_000 or max(n, m) >= 5000
        dtype = np.float32 if use_float32 else np.float64
        A_work = np.asarray(A, dtype=dtype, order="C")

        rng = np.random.default_rng(42)
        Omega = rng.standard_normal((m, l)).astype(dtype, copy=False)

        # Range finder
        Y = A_work @ Omega  # (n, l)
        for _ in range(n_iter):
            Y = A_work @ (A_work.T @ Y)

        # Orthonormalize and project
        Q, _ = np.linalg.qr(Y, mode="reduced")  # (n, l)
        B = Q.T @ A_work  # (l, m)

        # SVD on smaller matrix
        Ub, s_all, Vt = np.linalg.svd(B, full_matrices=False)

        U = Q @ Ub[:, :k]
        S = s_all[:k].copy()
        V = Vt[:k, :].T

        return {"U": np.asarray(U, dtype=float)[:, :k],
                "S": np.asarray(S, dtype=float)[:k],
                "V": np.asarray(V, dtype=float)[:, :k]}