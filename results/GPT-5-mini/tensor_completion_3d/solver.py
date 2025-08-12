import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        3D tensor completion using ADMM with SVT (singular value thresholding)
        applied to each mode-unfolding. Returns {"completed_tensor": ...}.
        """
        # Parse input
        try:
            M = np.array(problem["tensor"], dtype=float)
            mask = np.array(problem["mask"])
        except Exception:
            return {"completed_tensor": []}

        # If not 3D, attempt to reshape using tensor_dims
        if M.ndim != 3:
            dims = problem.get("tensor_dims")
            if dims is None:
                return {"completed_tensor": []}
            try:
                dims = tuple(int(x) for x in dims)
                if int(np.prod(dims)) != M.size:
                    return {"completed_tensor": []}
                M = M.reshape(dims)
            except Exception:
                return {"completed_tensor": []}

        # Normalize mask and ensure shapes match
        try:
            mask = np.array(mask, dtype=bool)
            if mask.size == M.size and mask.shape != M.shape:
                mask = mask.reshape(M.shape)
        except Exception:
            return {"completed_tensor": []}
        if mask.shape != M.shape:
            return {"completed_tensor": []}

        d1, d2, d3 = M.shape

        # Trivial cases
        if mask.all():
            return {"completed_tensor": M.tolist()}
        if not mask.any():
            return {"completed_tensor": np.zeros_like(M).tolist()}

        # Initialize X by filling missing entries with mean of observed
        X = M.copy()
        observed = M[mask]
        fill_val = float(np.mean(observed)) if observed.size > 0 else 0.0
        X[~mask] = fill_val

        # Precompute unfold/fold shapes
        shapes = [(d1, d2 * d3), (d2, d1 * d3), (d3, d1 * d2)]

        def unfold(T: np.ndarray, mode: int) -> np.ndarray:
            if mode == 0:
                return T.reshape(shapes[0])
            if mode == 1:
                return np.transpose(T, (1, 0, 2)).reshape(shapes[1])
            return np.transpose(T, (2, 0, 1)).reshape(shapes[2])

        def fold(A: np.ndarray, mode: int) -> np.ndarray:
            if mode == 0:
                return A.reshape(d1, d2, d3)
            if mode == 1:
                return np.transpose(A.reshape(d2, d1, d3), (1, 0, 2))
            return np.transpose(A.reshape(d3, d1, d2), (1, 2, 0))

        # Singular Value Thresholding
        def svt(A: np.ndarray, tau: float) -> np.ndarray:
            if tau <= 0:
                return A.copy()
            try:
                U, s, Vt = np.linalg.svd(A, full_matrices=False)
            except np.linalg.LinAlgError:
                return A.copy()
            s_shr = s - tau
            pos = s_shr > 0
            if not np.any(pos):
                return np.zeros_like(A)
            U_k = U[:, pos]
            Vt_k = Vt[pos, :]
            S_k = s_shr[pos]
            return (U_k * S_k) @ Vt_k

        # ADMM parameters (can be tuned)
        max_iter = int(kwargs.get("max_iter", 200))
        tol = float(kwargs.get("tol", 1e-4))
        weights = kwargs.get("weights", [1.0, 1.0, 1.0])
        try:
            weights = np.asarray(weights, dtype=float)
            if weights.size != 3:
                weights = np.ones(3, dtype=float)
        except Exception:
            weights = np.ones(3, dtype=float)
        rho = float(kwargs.get("rho", 1.0))

        # Initialize auxiliary and dual variables
        Y = [unfold(X, m).copy() for m in range(3)]
        Z = [np.zeros_like(Y[m]) for m in range(3)]
        num_modes = 3

        # Baseline for residual scaling
        norm_obs = np.linalg.norm(M[mask])
        if norm_obs == 0.0:
            norm_obs = 1.0

        for it in range(max_iter):
            X_prev = X.copy()

            # Y-update (SVT) for each unfolding
            for m in range(3):
                V = unfold(X, m) + (1.0 / rho) * Z[m]
                tau = weights[m] / rho
                Y[m] = svt(V, tau)

            # X-update: average of folded terms
            acc = np.zeros_like(X, dtype=float)
            for m in range(3):
                acc += fold(Y[m] - (1.0 / rho) * Z[m], m)
            X = acc / float(num_modes)

            # Enforce observed entries exactly
            X[mask] = M[mask]

            # Dual updates and residual calculation
            max_primal = 0.0
            for m in range(3):
                # primal residual = A(X) - Y
                diff = unfold(X, m) - Y[m]
                Z[m] = Z[m] + rho * diff
                val = np.linalg.norm(diff)
                if val > max_primal:
                    max_primal = val
            max_dual = rho * max(
                np.linalg.norm(unfold(X, m) - unfold(X_prev, m)) for m in range(3)
            )

            rel_primal = max_primal / (norm_obs + 1e-12)
            rel_dual = max_dual / (norm_obs + 1e-12)

            if rel_primal < tol and rel_dual < tol:
                break

            # Mild adaptive rho
            rho = min(rho * 1.01, 1e6)

        # Finalize and sanitize output
        X[mask] = M[mask]
        X = np.nan_to_num(X, nan=0.0, posinf=1e12, neginf=-1e12)

        return {"completed_tensor": X.tolist()}