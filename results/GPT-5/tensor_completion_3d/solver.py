from typing import Any, Dict

import numpy as np

class Solver:
    @staticmethod
    def _rand_svd(A: np.ndarray, k: int, p: int = 7, q: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Lightweight randomized SVD for dense matrices.
        Returns U, s, Vt with shapes (m, k), (k,), (k, n).
        """
        m, n = A.shape
        k = max(1, min(k, min(m, n)))
        # Random test matrix
        rng = np.random.default_rng()
        Omega = rng.standard_normal(size=(n, k + p), dtype=A.dtype)
        # Sample the range of A
        Y = A @ Omega
        # Power iterations to improve approximation
        for _ in range(max(0, q)):
            Y = A @ (A.T @ Y)
        # Orthonormal basis
        Q, _ = np.linalg.qr(Y, mode="reduced")
        # Project A to small matrix
        B = Q.T @ A  # (k+p) x n
        # SVD on small matrix
        Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
        # Map back
        U = Q @ Ub
        # Truncate to k components (or available)
        k_eff = min(k, U.shape[1], Vt.shape[0])
        return U[:, :k_eff], s[:k_eff], Vt[:k_eff, :]

    def _svt(self, W: np.ndarray, tau: float) -> np.ndarray:
        """
        Singular value thresholding: prox operator for nuclear norm.
        Returns argmin_Z 0.5*||Z - W||_F^2 + tau*||Z||_*.
        Uses a custom randomized SVD for larger matrices to accelerate computation.
        """
        m, n = W.shape
        mn = min(m, n)

        # Small problems: full SVD is typically faster and more accurate
        if mn < 100:
            U, s, Vt = np.linalg.svd(W, full_matrices=False)
            s_shrink = s - tau
            idx = s_shrink > 0.0
            if not np.any(idx):
                return np.zeros_like(W)
            return (U[:, idx] * s_shrink[idx]) @ Vt[idx, :]

        # Larger problems: randomized SVD with adaptive rank
        k = min(32, mn - 1) if mn > 1 else 1
        if k < 1:
            return np.zeros_like(W)

        # Try a couple of increasing ranks until smallest retained singular <= tau
        while True:
            try:
                U, s, Vt = self._rand_svd(W, k=k, p=7, q=1)
            except Exception:
                # Fallback to full SVD
                U, s, Vt = np.linalg.svd(W, full_matrices=False)
            if s.size == 0:
                return np.zeros_like(W)
            if s[-1] <= tau or k >= mn - 1:
                break
            k = min(k * 2, mn - 1)

        s_shrink = s - tau
        idx = s_shrink > 0.0
        if not np.any(idx):
            return np.zeros_like(W)
        return (U[:, idx] * s_shrink[idx]) @ Vt[idx, :]

    def _admm_matrix_completion(
        self,
        M: np.ndarray,
        mask: np.ndarray,
        rho: float = 1.0,
        rel_tol: float = 1e-3,
        abs_tol: float = 1e-4,
        max_iter: int = 500,
        adapt_rho: bool = True,
        over_relaxation: float = 1.8,
    ) -> np.ndarray:
        """
        Solve: minimize ||Z||_* subject to P_Omega(X) = P_Omega(M), X = Z
        using ADMM on the mode-1 unfolding matrix.

        Returns X (which equals Z at convergence).
        """
        M = np.asarray(M, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        m, n = M.shape
        N = m * n
        sqrtN = np.sqrt(N)

        # Initialize variables with warm start
        X = np.zeros((m, n), dtype=np.float64)
        if mask.any():
            X[mask] = M[mask]
        try:
            Z = self._svt(X, 1.0 / rho)
        except np.linalg.LinAlgError:
            Z = np.zeros_like(X)
        U = X - Z

        # Precompute indices of observed entries for faster assignment
        obs_idx = np.nonzero(mask) if mask.any() else None

        # ADMM parameters for adaptive rho
        mu = 10.0
        tau_incr = 2.0
        tau_decr = 2.0

        Z_prev = Z.copy()

        alpha = float(over_relaxation)
        if not (1.0 <= alpha <= 1.9):
            alpha = 1.8

        # Iterate
        for it in range(max_iter):
            # X-update: projection onto equality constraints for observed entries
            X = Z - U
            if obs_idx is not None:
                X[obs_idx] = M[obs_idx]

            # Over-relaxation for improved convergence
            X_hat = alpha * X + (1.0 - alpha) * Z

            # Z-update: SVT on (X_hat + U) with threshold 1/rho
            W = X_hat + U
            try:
                Z = self._svt(W, 1.0 / rho)
            except np.linalg.LinAlgError:
                # Keep previous Z if SVD fails
                Z = Z

            # U-update (scaled dual)
            U = U + (X_hat - Z)

            # Stopping criteria (primal and dual residuals)
            r = X - Z
            s = rho * (Z - Z_prev)

            r_norm = np.linalg.norm(r, ord="fro")
            s_norm = np.linalg.norm(s, ord="fro")

            X_norm = np.linalg.norm(X, ord="fro")
            Z_norm = np.linalg.norm(Z, ord="fro")
            U_norm = np.linalg.norm(U, ord="fro")

            eps_pri = abs_tol * sqrtN + rel_tol * max(X_norm, Z_norm)
            eps_dual = abs_tol * sqrtN + rel_tol * (rho * U_norm)

            if r_norm <= eps_pri and s_norm <= eps_dual:
                break

            # Adapt rho to balance residuals (every few iterations to avoid jitter)
            if adapt_rho and (it % 5 == 0):
                rho_new = rho
                if r_norm > mu * s_norm:
                    rho_new = rho * tau_incr
                elif s_norm > mu * r_norm:
                    rho_new = rho / tau_decr

                if rho_new != rho and rho_new > 0:
                    # Scale dual variable when rho changes (scaled form)
                    U *= rho / rho_new
                    rho = rho_new

            Z_prev = Z

        # Ensure exact data fidelity on observed entries
        if obs_idx is not None:
            X[obs_idx] = M[obs_idx]
        return X

    def solve(self, problem: Dict, **kwargs) -> Dict[str, Any]:
        """
        Solve the 3D tensor completion problem via nuclear norm minimization
        on the mode-1 unfolding using a fast ADMM-based SVT algorithm.

        :param problem: Dictionary with keys "tensor", "mask", "tensor_dims"
        :return: {"completed_tensor": 3D list}
        """
        try:
            observed_tensor = np.asarray(problem["tensor"], dtype=np.float64)
            mask = np.asarray(problem["mask"], dtype=bool)

            # Validate dimensions
            if observed_tensor.ndim != 3 or mask.shape != observed_tensor.shape:
                return {"completed_tensor": []}

            dim1, dim2, dim3 = observed_tensor.shape

            # Edge cases
            if mask.all():
                return {"completed_tensor": observed_tensor.tolist()}
            if not mask.any():
                return {"completed_tensor": np.zeros_like(observed_tensor).tolist()}

            # Mode-1 unfolding
            unfolding1 = observed_tensor.reshape(dim1, dim2 * dim3)
            mask1 = mask.reshape(dim1, dim2 * dim3)

            # ADMM parameters (allow overrides via kwargs)
            rho = float(kwargs.get("rho", 1.0))
            rel_tol = float(kwargs.get("rel_tol", 1e-3))
            abs_tol = float(kwargs.get("abs_tol", 1e-4))
            max_iter = int(kwargs.get("max_iter", 500))
            adapt_rho = bool(kwargs.get("adapt_rho", True))
            over_relaxation = float(kwargs.get("over_relaxation", 1.8))

            # Solve matrix completion on mode-1 unfolding
            X_opt = self._admm_matrix_completion(
                M=unfolding1,
                mask=mask1,
                rho=rho,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                max_iter=max_iter,
                adapt_rho=adapt_rho,
                over_relaxation=over_relaxation,
            )

            # Fold back into tensor shape
            completed_tensor = X_opt.reshape((dim1, dim2, dim3))

            # Enforce exact fidelity at observed entries (numerical safety)
            if mask.any():
                completed_tensor[mask] = observed_tensor[mask]

            # Replace any NaNs or infs
            if not np.isfinite(completed_tensor).all():
                completed_tensor = np.nan_to_num(completed_tensor, copy=False)

            return {"completed_tensor": completed_tensor.tolist()}

        except Exception:
            return {"completed_tensor": []}