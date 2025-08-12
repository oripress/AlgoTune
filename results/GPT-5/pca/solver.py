from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> np.ndarray:
        """
        Fast PCA using eigen-decomposition of the smaller matrix:
        - If m >= n: eigen-decomposition of C = X^T X (n x n)
        - Else: eigen-decomposition of G = X X^T (m x m) and map to right singular vectors
        Returns V of shape (k, n) with orthonormal rows.
        """
        X = np.asarray(problem["X"], dtype=np.float64)
        m, n = X.shape
        k_req = int(problem["n_components"])

        if k_req <= 0:
            return np.empty((0, n), dtype=np.float64)

        # If requesting a full orthonormal basis in R^n, any orthonormal matrix suffices (objective is ||X||_F^2).
        if k_req == n:
            return np.eye(n, dtype=np.float64)

        # Center data
        mu = X.mean(axis=0)
        X = X - mu

        # Helper: complete an orthonormal basis to size t columns, given existing Q (n x q) with orthonormal columns
        def complete_basis(Q: np.ndarray, t: int, n_dim: int) -> np.ndarray:
            if t <= 0:
                return np.empty((n_dim, 0), dtype=np.float64)
            if Q.size == 0:
                return np.eye(n_dim, t, dtype=np.float64)
            comp_cols = []
            QT = Q.T
            for j in range(n_dim):
                if len(comp_cols) >= t:
                    break
                v = np.zeros(n_dim, dtype=np.float64)
                v[j] = 1.0
                v -= Q @ (QT @ v)
                for u in comp_cols:
                    v -= u * (u @ v)
                nv = np.linalg.norm(v)
                if nv > 1e-12:
                    comp_cols.append(v / nv)
            if len(comp_cols) < t:
                rem = t - len(comp_cols)
                Z = np.random.default_rng(0).standard_normal((n_dim, rem))
                Z -= Q @ (QT @ Z)
                if comp_cols:
                    U = np.stack(comp_cols, axis=1)
                    Z -= U @ (U.T @ Z)
                Q2, _ = np.linalg.qr(Z, mode="reduced")
                comp_cols.extend([Q2[:, i] for i in range(min(rem, Q2.shape[1]))])
            if not comp_cols:
                return np.eye(n_dim, t, dtype=np.float64)
            return np.stack(comp_cols, axis=1)[:, :t]

        # Early exit for wide data when requesting at least m components:
        if m < n and k_req >= m:
            Q, _ = np.linalg.qr(X.T, mode="reduced")  # Q: (n x m), columns orthonormal spanning row space
            if k_req == m:
                return Q.T
            extra = complete_basis(Q, k_req - m, n)  # (n x (k_req - m))
            return np.vstack((Q.T, extra.T))

        r = min(m, n)
        k_eff = min(k_req, r)

        if m >= n:
            # Use covariance matrix (n x n)
            C = X.T @ X
            w, V = np.linalg.eigh(C)  # ascending eigenvalues
            V_k_cols = V[:, -k_eff:]  # take top-k columns (largest eigenvalues)
            V_rows = V_k_cols.T
            if k_req > k_eff:
                V_extra_cols = complete_basis(V_k_cols, k_req - k_eff, n)
                V_rows = np.vstack((V_rows, V_extra_cols.T))
            return V_rows

        # m < n and k_req < m: use Gram matrix (m x m) eigendecomposition
        G = X @ X.T
        w, U = np.linalg.eigh(G)  # ascending
        U_k = U[:, -k_eff:]
        w_k = w[-k_eff:]

        # Map to right singular vectors: V = X^T U / sqrt(w)
        eps = max((w_k[-1] if w_k.size else 0.0) * 1e-12, 1e-15) if w_k.size else 0.0
        pos = w_k > eps
        if np.any(pos):
            s_pos = np.sqrt(w_k[pos])
            U_pos = U_k[:, pos]
            V_cols = (X.T @ U_pos) / s_pos[None, :]
        else:
            V_cols = np.empty((n, 0), dtype=np.float64)

        # Complete to k_eff columns if rank-deficient
        r1 = V_cols.shape[1]
        if r1 < k_eff:
            V_extra = complete_basis(V_cols, k_eff - r1, n)  # (n x (k_eff - r1))
            V_cols = np.hstack((V_cols, V_extra))
        V_rows = V_cols.T  # (k_eff x n)

        if k_req > k_eff:
            V_extra2 = complete_basis(V_cols, k_req - k_eff, n)
            V_rows = np.vstack((V_rows, V_extra2.T))
        return V_rows