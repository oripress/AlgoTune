from typing import Any, Dict
import numpy as np
import scipy.linalg as la

class Solver:
    def __init__(self):
        # Maximum dimension (max(n,m)) to attempt the eigen-based approach.
        # For larger sizes, fall back to LAPACK-backed solver which is typically more stable.
        self.max_eig_dim = 300
        # Tolerance for detecting (near-)Hermitian / symmetric matrices
        self.hermitian_tol = 1e-12

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the Sylvester equation A X + X B = Q.

        Strategy:
        - Fast scalar (1x1) path.
        - For moderate sizes (controlled by max_eig_dim) attempt eigen-based solver:
          * If A and B are (near-)Hermitian use eigh (unitary eigenvectors).
          * Otherwise use eig and linear solves to avoid explicit inverses.
        - On any failure or for large sizes, fall back to scipy.linalg.solve_sylvester.
        """
        A = np.asarray(problem["A"])
        B = np.asarray(problem["B"])
        Q = np.asarray(problem["Q"])

        # Basic validation
        if A.ndim != 2 or B.ndim != 2 or Q.ndim != 2:
            raise ValueError("A, B, Q must be 2-D arrays")
        n, n2 = A.shape
        m, m2 = B.shape
        if n != n2 or m != m2:
            raise ValueError("A and B must be square matrices")
        if Q.shape != (n, m):
            raise ValueError(f"Q must have shape ({n}, {m}), got {Q.shape}")

        # 1x1 fast path
        if n == 1 and m == 1:
            denom = A.item() + B.item()
            X = Q / denom
            return {"X": X}

        # Determine whether inputs are complex
        inputs_complex = np.iscomplexobj(A) or np.iscomplexobj(B) or np.iscomplexobj(Q)
        max_eig_dim = int(kwargs.get("max_eig_dim", self.max_eig_dim))
        try_eig = max(n, m) <= max_eig_dim

        if try_eig:
            try:
                dtype = np.complex128 if inputs_complex else np.float64
                A_c = np.asfortranarray(A, dtype=dtype)
                B_c = np.asfortranarray(B, dtype=dtype)
                Q_c = np.asfortranarray(Q, dtype=dtype)

                # Detect (near-)Hermitian / symmetric matrices
                if inputs_complex:
                    is_A_herm = np.allclose(A_c, A_c.conj().T, atol=self.hermitian_tol, rtol=0.0)
                    is_B_herm = np.allclose(B_c, B_c.conj().T, atol=self.hermitian_tol, rtol=0.0)
                else:
                    is_A_herm = np.allclose(A_c, A_c.T, atol=self.hermitian_tol, rtol=0.0)
                    is_B_herm = np.allclose(B_c, B_c.T, atol=self.hermitian_tol, rtol=0.0)

                if is_A_herm and is_B_herm:
                    # Hermitian/symmetric fast path using eigh (unitary transforms)
                    alpha, VA = la.eigh(A_c)
                    beta, VB = la.eigh(B_c)

                    # Transform Q into eigen-bases
                    C = VA.conj().T.dot(Q_c).dot(VB)
                    denom = alpha[:, None] + beta[None, :]
                    eps = np.finfo(float).eps
                    if np.any(np.abs(denom) < 1e3 * eps):
                        # Dangerously small denominators; fallback
                        raise la.LinAlgError("Small denominators in hermitian eigen-based solve.")

                    Y = C / denom
                    X_c = VA.dot(Y).dot(VB.conj().T)
                else:
                    # General eigen-decomposition
                    alpha, VA = la.eig(A_c)
                    beta, VB = la.eig(B_c)

                    # Compute C = VA^{-1} * Q * VB efficiently (solve linear systems)
                    QVB = Q_c.dot(VB)
                    C = la.solve(VA, QVB)

                    denom = alpha[:, None] + beta[None, :]
                    eps = np.finfo(float).eps
                    if np.any(np.abs(denom) < 1e3 * eps):
                        raise la.LinAlgError("Small denominators in eigen-based solve; fallback requested.")

                    Y = C / denom

                    # Recover X = VA * Y * VB^{-1} without explicitly inverting VB
                    S = VA.dot(Y)
                    # Solve VB^T * Z = S^T  => X = Z^T
                    X_c = la.solve(VB.T, S.T).T

                # If inputs were real but numerical work produced tiny imaginary parts, cast to real
                if not inputs_complex and np.iscomplexobj(X_c):
                    imag_max = float(np.max(np.abs(X_c.imag)))
                    real_scale = float(max(1.0, np.max(np.abs(X_c.real))))
                    if imag_max <= 1e-12 * real_scale:
                        X = X_c.real
                    else:
                        X = X_c
                else:
                    X = X_c

                return {"X": X}
            except Exception:
                # Any failure in eigen-path -> fall back to LAPACK solver
                pass

        # Robust fallback using LAPACK-backed solver
        dtype = np.complex128 if inputs_complex else np.float64
        A_f = np.asfortranarray(A, dtype=dtype)
        B_f = np.asfortranarray(B, dtype=dtype)
        Q_f = np.asfortranarray(Q, dtype=dtype)

        X = la.solve_sylvester(A_f, B_f, Q_f)

        # Cast back to real if inputs were real but tiny imaginary parts exist
        if not inputs_complex and np.iscomplexobj(X):
            imag_max = float(np.max(np.abs(X.imag)))
            real_scale = float(max(1.0, np.max(np.abs(X.real))))
            if imag_max <= 1e-12 * real_scale:
                X = X.real

        return {"X": X}