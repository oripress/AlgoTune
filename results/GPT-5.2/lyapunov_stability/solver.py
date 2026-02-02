from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import LinAlgError, solve_discrete_lyapunov

def _try_fast_lyapunov_certificate(A: np.ndarray) -> tuple[bool, np.ndarray | None]:
    """
    Construct P_K = sum_{k=0}^{K} (A^k)^T (A^k).
    With K = 2^m - 1, we have:
        A^T P_K A - P_K = (A^(K+1))^T (A^(K+1)) - I

    If I - (A^(K+1))^T (A^(K+1)) is PD, then the system is stable and P_K is a valid
    Lyapunov matrix. We compute P_K efficiently via doubling.
    """
    n = A.shape[0]
    I = np.eye(n, dtype=A.dtype)

    P = I.copy()
    B = A.copy()  # after m steps, B = A^(2^m)

    # Choose a small set of targets to keep runtime predictable.
    if n <= 4:
        targets = (4, 7, 10, 13)  # exponents 16..8192
    elif n <= 12:
        targets = (3, 6, 9, 12)  # exponents 8..4096
    else:
        targets = (2, 4, 6, 8)  # exponents 4..256

    step = 0
    for target in targets:
        while step < target:
            # P <- P + B^T P B
            PB = P @ B
            P += B.T @ PB

            # B <- B^2
            B = B @ B
            step += 1

        # Only check finiteness at checkpoints (we only accept certificates here).
        if not (np.isfinite(B).all() and np.isfinite(P).all()):
            return False, None

        # Check stability certificate: I - B^T B ≻ 0  (i.e., ||A^(2^step)||_2 < 1)
        M = I - (B.T @ B)
        try:
            np.linalg.cholesky(M)
        except LinAlgError:
            continue

        P = (P + P.T) * 0.5
        return True, P

    return False, None

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        A = np.asarray(problem["A"], dtype=np.float64)
        n = A.shape[0]
        if n == 0:
            return {"is_stable": True, "P": []}

        ok, P = _try_fast_lyapunov_certificate(A)
        if ok and P is not None:
            return {"is_stable": True, "P": P.tolist()}

        # Robust fallback: solve (A^T) P A - P = -I and validate quickly.
        I = np.eye(n, dtype=A.dtype)
        try:
            P2 = solve_discrete_lyapunov(A.T, I)
        except Exception:
            return {"is_stable": False, "P": None}

        P2 = (P2 + P2.T) * 0.5
        try:
            np.linalg.cholesky(P2)
        except LinAlgError:
            return {"is_stable": False, "P": None}

        S = A.T @ P2 @ A - P2
        S = (S + S.T) * 0.5
        try:
            np.linalg.cholesky(-S)
        except LinAlgError:
            return {"is_stable": False, "P": None}

        return {"is_stable": True, "P": P2.tolist()}