from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_are

class Solver:
    def __init__(self) -> None:
        self._tail_tol = 1e-12
        self._power_steps = 10

    def _finite_lyapunov_certificate(self, A: np.ndarray) -> np.ndarray | None:
        n = A.shape[0]
        if float(np.sum(A * A)) < 1.0 - self._tail_tol:
            return np.eye(n)

        P = np.eye(n)
        M = A.copy()
        for _ in range(self._power_steps):
            P = P + M.T @ M
            M = M @ A
            if float(np.sum(M * M)) < 1.0 - self._tail_tol:
                return 0.5 * (P + P.T)
        return None

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        try:
            A = np.asarray(problem["A"], dtype=float)
            B = np.asarray(problem["B"], dtype=float)

            if A.ndim != 2 or B.ndim != 2:
                return {"is_stabilizable": False, "K": None, "P": None}

            n = A.shape[0]
            if A.shape[1] != n or B.shape[0] != n:
                return {"is_stabilizable": False, "K": None, "P": None}

            m = B.shape[1]

            if n == 1:
                a = float(A[0, 0])
                if abs(a) < 1.0 - self._tail_tol:
                    return {
                        "is_stabilizable": True,
                        "K": np.zeros((m, 1), dtype=float),
                        "P": np.array([[1.0]], dtype=float),
                    }
                brow = B[0]
                j = int(np.argmax(np.abs(brow)))
                bj = float(brow[j])
                if abs(bj) <= self._tail_tol:
                    return {"is_stabilizable": False, "K": None, "P": None}
                K = np.zeros((m, 1), dtype=float)
                K[j, 0] = -a / bj
                return {
                    "is_stabilizable": True,
                    "K": K,
                    "P": np.array([[1.0]], dtype=float),
                }

            P = self._finite_lyapunov_certificate(A)
            if P is not None and np.all(np.isfinite(P)):
                return {"is_stabilizable": True, "K": np.zeros((m, n), dtype=float), "P": P}

            eye_n = np.eye(n)

            if m == 1:
                b = B[:, 0]
                btb = float(b @ b)
                bta = b @ A

                for q in (1.0, 10.0, 100.0):
                    g = 1.0 + q * btb
                    Krow = -(q / g) * bta
                    Acl = A + np.outer(b, Krow)
                    P = self._finite_lyapunov_certificate(Acl)
                    if P is not None:
                        return {"is_stabilizable": True, "K": Krow[None, :], "P": P}

                for q in (1.0, 10.0):
                    X = q * eye_n
                    for _ in range(12):
                        Xb = X @ b
                        g = 1.0 + float(b @ Xb)
                        F = b @ X @ A
                        Krow = -(1.0 / g) * F
                        Acl = A + np.outer(b, Krow)
                        P = self._finite_lyapunov_certificate(Acl)
                        if P is not None:
                            return {"is_stabilizable": True, "K": Krow[None, :], "P": P}
                        Y = (1.0 / g) * F
                        X = A.T @ X @ A - np.outer(F, Y) + q * eye_n

                X = solve_discrete_are(A, B, eye_n, np.eye(1), balanced=False)
                X = np.asarray(np.real_if_close(X, tol=1000), dtype=float)
                X = 0.5 * (X + X.T)
                Xb = X @ b
                g = 1.0 + float(b @ Xb)
                F = b @ X @ A
                Krow = -(1.0 / g) * F
                K = Krow[None, :]
                Acl = A + np.outer(b, Krow)
            else:
                eye_m = np.eye(m)
                Bt = B.T
                BtB = Bt @ B
                BtA = Bt @ A

                for q in (1.0, 10.0, 100.0):
                    G = eye_m + q * BtB
                    F = q * BtA
                    K = -np.linalg.solve(G, F)
                    Acl = A + B @ K
                    P = self._finite_lyapunov_certificate(Acl)
                    if P is not None and np.all(np.isfinite(K)):
                        return {"is_stabilizable": True, "K": K, "P": P}

                for q in (1.0, 10.0):
                    X = q * eye_n
                    for _ in range(12):
                        XB = X @ B
                        G = eye_m + Bt @ XB
                        F = Bt @ X @ A
                        K = -np.linalg.solve(G, F)
                        Acl = A + B @ K
                        P = self._finite_lyapunov_certificate(Acl)
                        if P is not None and np.all(np.isfinite(K)):
                            return {"is_stabilizable": True, "K": K, "P": P}
                        Y = np.linalg.solve(G, F)
                        X = A.T @ X @ A - F.T @ Y + q * eye_n

                X = solve_discrete_are(A, B, eye_n, eye_m, balanced=False)
                X = np.asarray(np.real_if_close(X, tol=1000), dtype=float)
                X = 0.5 * (X + X.T)
                G = eye_m + Bt @ X @ B
                F = Bt @ X @ A
                K = -np.linalg.solve(G, F)
                K = np.asarray(np.real_if_close(K, tol=1000), dtype=float)
                Acl = A + B @ K

            S = Acl.T @ X @ Acl - X
            S = 0.5 * (S + S.T)

            if not np.all(np.isfinite(K)) or not np.all(np.isfinite(X)):
                return {"is_stabilizable": False, "K": None, "P": None}
            if float(np.min(np.linalg.eigvalsh(X))) <= 1e-10:
                return {"is_stabilizable": False, "K": None, "P": None}
            if float(np.min(np.linalg.eigvalsh(-S))) <= 1e-10:
                return {"is_stabilizable": False, "K": None, "P": None}
            return {"is_stabilizable": True, "K": K, "P": X}
        except Exception:
            return {"is_stabilizable": False, "K": None, "P": None}