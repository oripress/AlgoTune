from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Fast NMF solver using NNDSVD-like initialization (when affordable)
        and vectorized multiplicative updates.

        Inputs in `problem`:
            - "X": 2D array-like (m x n)
            - "n_components": int, rank k

        Returns:
            {"W": W.tolist(), "H": H.tolist()} where W is (m x k) and H is (k x n)
        """
        try:
            X_in = problem.get("X", None)
            if X_in is None:
                return {"W": [], "H": []}

            X = np.array(X_in, dtype=float)

            # Handle degenerate / empty cases
            if X.size == 0:
                k0 = int(problem.get("n_components", 0) or 0)
                m0 = X.shape[0] if X.ndim >= 1 else 0
                return {"W": np.zeros((m0, k0)).tolist(), "H": np.zeros((k0, 0)).tolist()}

            if X.ndim == 1:
                X = X.reshape((-1, 1))
            m, n = X.shape

            k = int(problem.get("n_components", min(m, n)))
            if k <= 0:
                return {"W": np.zeros((m, 0)).tolist(), "H": np.zeros((0, n)).tolist()}

            # Enforce non-negativity for X
            X = np.maximum(X, 0.0)

            rng = np.random.default_rng(0)
            eps = 1e-12

            # Initialize W, H
            W = None
            H = None

            # Use NNDSVD-like initialization when SVD affordable
            try:
                if m * n <= 200000:
                    U, S, Vt = np.linalg.svd(X, full_matrices=False)
                    r = min(k, S.size)
                    W = np.zeros((m, k), dtype=float)
                    H = np.zeros((k, n), dtype=float)
                    for j in range(r):
                        uj = U[:, j]
                        vj = Vt[j, :]
                        sj = float(S[j])

                        uj_p = np.maximum(uj, 0.0)
                        uj_n = np.maximum(-uj, 0.0)
                        vj_p = np.maximum(vj, 0.0)
                        vj_n = np.maximum(-vj, 0.0)

                        nj_p = np.linalg.norm(uj_p)
                        nj_n = np.linalg.norm(uj_n)
                        mj_p = np.linalg.norm(vj_p)
                        mj_n = np.linalg.norm(vj_n)

                        if nj_p * mj_p >= nj_n * mj_n:
                            if nj_p <= eps or mj_p <= eps:
                                W[:, j] = rng.random(m) * 1e-6 + eps
                                H[j, :] = rng.random(n) * 1e-6 + eps
                            else:
                                coef = np.sqrt(max(sj, 0.0))
                                W[:, j] = coef * (uj_p / nj_p)
                                H[j, :] = coef * (vj_p / mj_p)
                        else:
                            if nj_n <= eps or mj_n <= eps:
                                W[:, j] = rng.random(m) * 1e-6 + eps
                                H[j, :] = rng.random(n) * 1e-6 + eps
                            else:
                                coef = np.sqrt(max(sj, 0.0))
                                W[:, j] = coef * (uj_n / nj_n)
                                H[j, :] = coef * (vj_n / mj_n)

                    if k > r:
                        W[:, r:k] = rng.random((m, k - r)) * 1e-6 + eps
                        H[r:k, :] = rng.random((k - r, n)) * 1e-6 + eps
                else:
                    # too large, fall back to random init
                    raise RuntimeError("Skip SVD for large matrix")
            except Exception:
                W = rng.random((m, k)).astype(float) * 1e-2 + eps
                H = rng.random((k, n)).astype(float) * 1e-2 + eps

            W = np.maximum(W, eps)
            H = np.maximum(H, eps)

            # Fix any degenerate components
            deg = ((W.sum(axis=0) <= eps) | (H.sum(axis=1) <= eps))
            if deg.any():
                for j in np.where(deg)[0]:
                    W[:, j] = rng.random(m) * 1e-3 + eps
                    H[j, :] = rng.random(n) * 1e-3 + eps

            # Adaptive iteration budget
            size = m * n
            if size <= 2000:
                max_iter = 500
            elif size <= 20000:
                max_iter = 250
            elif size <= 200000:
                max_iter = 120
            else:
                max_iter = 60

            prev_err = None
            tol_rel = 1e-7

            # Multiplicative updates
            for it in range(int(max_iter)):
                # Update H
                WH = W.T @ X
                denom = (W.T @ W) @ H + eps
                H *= WH / denom
                H = np.maximum(H, eps)

                # Update W
                XHT = X @ H.T
                denomW = W @ (H @ H.T) + eps
                W *= XHT / denomW
                W = np.maximum(W, eps)

                # Check convergence periodically
                if (it & 7) == 0:
                    WH_full = W @ H
                    err = 0.5 * float(np.linalg.norm(X - WH_full) ** 2)
                    if prev_err is not None:
                        if prev_err - err <= tol_rel * max(1.0, prev_err):
                            break
                    prev_err = err

            return {"W": W.tolist(), "H": H.tolist()}

        except Exception:
            # fallback: zero matrices with correct shapes
            try:
                Xf = np.array(problem.get("X", [[0.0]]), dtype=float)
                if Xf.ndim == 1:
                    Xf = Xf.reshape((-1, 1))
                m0, n0 = Xf.shape
                k0 = int(problem.get("n_components", min(m0, n0)))
                W0 = np.zeros((m0, k0), dtype=float)
                H0 = np.zeros((k0, n0), dtype=float)
                return {"W": W0.tolist(), "H": H0.tolist()}
            except Exception:
                return {"W": [], "H": []}