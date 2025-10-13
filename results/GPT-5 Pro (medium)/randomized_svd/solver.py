from typing import Any
import numpy as np


class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Fast Randomized SVD solver.

        Inputs:
          problem: dict with keys
            - "matrix": array-like shape (n, m)
            - "n_components": int
            - "matrix_type": str (optional) - influences number of power iterations

        Outputs:
          dict with keys:
            - "U": np.ndarray, shape (n, k)
            - "S": np.ndarray, shape (k,)
            - "V": np.ndarray, shape (m, k)
        """
        A = np.asarray(problem["matrix"], dtype=float)
        k: int = int(problem["n_components"])
        n, m = A.shape
        matrix_type = problem.get("matrix_type", "generic")
        r = min(n, m)

        if k <= 0:
            return {
                "U": np.zeros((n, 0), dtype=float),
                "S": np.zeros((0,), dtype=float),
                "V": np.zeros((m, 0), dtype=float),
            }

        # If requesting near/full rank or small problems, compute exact SVD for accuracy/simplicity.
        if k >= r or r <= 64:
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            k2 = min(k, s.shape[0])
            U = U[:, :k2]
            s = s[:k2]
            V = Vt[:k2, :].T
            # If k > available rank (shouldn't happen in valid tasks), extend with orthonormal columns and zero singulars.
            if k2 < k:
                rng = np.random.default_rng(42)
                # Extend U
                tU = k - k2
                GU = rng.standard_normal((n, tU))
                if k2 > 0:
                    GU -= U @ (U.T @ GU)
                QU, _ = np.linalg.qr(GU, mode="reduced")
                U = np.concatenate([U, QU[:, :tU]], axis=1)
                # Extend V
                tV = k - k2
                GV = rng.standard_normal((m, tV))
                if k2 > 0:
                    GV -= V @ (V.T @ GV)
                QV, _ = np.linalg.qr(GV, mode="reduced")
                V = np.concatenate([V, QV[:, :tV]], axis=1)
                s = np.concatenate([s, np.zeros(k - k2, dtype=s.dtype)], axis=0)
            return {"U": U, "S": s, "V": V}

        # Number of power iterations (stabilized) depending on matrix type
        n_iter = 10 if matrix_type == "ill_conditioned" else 5

        # Oversampling parameter with cap
        p = 10
        if k + p > r:
            p = max(2, r - k)
        l = k + p

        rng = np.random.default_rng(42)

        # Random test matrix and sketch
        Omega = rng.standard_normal((m, l))
        Y = A @ Omega  # (n x l)

        # Stabilized power iterations (Halko et al.)
        for _ in range(max(0, n_iter)):
            Y, _ = np.linalg.qr(Y, mode="reduced")
            Z = A.T @ Y  # (m x l)
            Z, _ = np.linalg.qr(Z, mode="reduced")
            Y = A @ Z  # (n x l)

        # Final orthonormal basis
        Q, _ = np.linalg.qr(Y, mode="reduced")  # (n x l)

        # Small matrix SVD
        B = Q.T @ A  # (l x m)
        Ub, s, Vt = np.linalg.svd(B, full_matrices=False)

        # Map back and truncate
        U_approx = Q @ Ub  # (n x l)
        U_k = U_approx[:, :k]
        s_k = s[:k]
        V_k = Vt[:k, :].T  # (m x k)

        return {"U": U_k, "S": s_k, "V": V_k}