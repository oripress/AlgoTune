import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        # Load covariance
        A = np.array(problem["covariance"], dtype=float, copy=False)
        n = A.shape[0]
        k = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])
        # number of non-zero components
        k0 = min(k, n)
        if k0 > 0:
            # ARPACK-based eigendecomposition with light convergence parameters
            if k0 < n:
                try:
                    eigvals, eigvecs = eigsh(
                        A, k=k0, which='LA',
                        tol=0.1, maxiter=5,
                        ncv=min(n, k0 * 2 + 1)
                    )
                    # sort descending
                    order = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[order]
                    eigvecs = eigvecs[:, order]
                except Exception:
                    # fallback to full symmetric eigen decomposition
                    w, V = np.linalg.eigh(A)
                    idx = np.argsort(w)[::-1][:k0]
                    eigvals = w[idx]
                    eigvecs = V[:, idx]
            else:
                # full eigen when k0 == n
                w, V = np.linalg.eigh(A)
                order = np.argsort(w)[::-1]
                eigvals = w[order]
                eigvecs = V[:, order]
            # clamp negatives and build B = U * sqrt(lambda)
            np.clip(eigvals, 0.0, None, out=eigvals)
            B = eigvecs * np.sqrt(eigvals)[None, :]
            # soft-threshold (L1 prox)
            thr = 0.5 * sparsity_param
            B = np.sign(B) * np.maximum(np.abs(B) - thr, 0.0)
            # enforce unit L2 norm per component
            norms = np.linalg.norm(B, axis=0)
            big = norms > 1.0
            if big.any():
                B[:, big] /= norms[big]
            # explained variance: sum_i λ_i * (u_i^T x)^2
            Y = eigvecs.T.dot(B)
            ev = np.sum((Y * Y) * eigvals[:, None], axis=0)
        else:
            B = np.zeros((n, 0), dtype=float)
            ev = np.empty((0,), dtype=float)
        # assemble full components matrix and pad zeros if k > k0
        X = np.zeros((n, k), dtype=float)
        if k0 > 0:
            X[:, :k0] = B
        explained_variance = ev.tolist() + [0.0] * (k - k0)
        return {"components": X.tolist(), "explained_variance": explained_variance}