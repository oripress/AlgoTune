from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import eigvals, solve_discrete_are, solve_discrete_lyapunov

# Tiny caches to avoid repeated np.eye allocations (helps when many instances share sizes).
_EYE_N: dict[int, np.ndarray] = {}
_EYE_M: dict[int, np.ndarray] = {}

def _eye_n(n: int) -> np.ndarray:
    E = _EYE_N.get(n)
    if E is None or E.shape[0] != n:
        E = np.eye(n)
        _EYE_N[n] = E
    return E

def _eye_m(m: int) -> np.ndarray:
    E = _EYE_M.get(m)
    if E is None or E.shape[0] != m:
        E = np.eye(m)
        _EYE_M[m] = E
    return E

def _pbh_stabilizable(A: np.ndarray, B: np.ndarray) -> bool:
    """
    PBH stabilizability test (discrete-time):
      for all eigenvalues |λ| >= 1, rank([λI - A, B]) = n.
    """
    n = A.shape[0]
    lam = np.linalg.eigvals(A)
    # Only check potentially problematic modes.
    mask = np.abs(lam) >= 1.0 - 1e-10
    if not np.any(mask):
        return True

    A_c = A.astype(complex, copy=False)
    B_c = B.astype(complex, copy=False)
    I = np.eye(n, dtype=complex)

    # De-duplicate close eigenvalues to reduce SVD calls.
    checked: list[complex] = []
    for l in lam[mask]:
        if any(abs(l - c) <= 1e-7 for c in checked):
            continue
        checked.append(l)
        M = np.hstack((l * I - A_c, B_c))
        s = np.linalg.svd(M, compute_uv=False)
        # Full row rank iff smallest singular value is not ~0.
        tol = max(1e-10, 1e-8 * float(s[0]))
        if float(s[-1]) <= tol:
            return False
    return True

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)

        n = int(A.shape[0])
        m = int(B.shape[1]) if B.ndim == 2 else 1

        In = _eye_n(n)
        eps = 1.0 - 1e-12

        # No control: stabilizable iff already Schur stable.
        if m == 0:
            # Cheap sufficient condition first
            if float(np.linalg.norm(A, ord=np.inf)) < eps:
                P0 = solve_discrete_lyapunov(A.T, In)
                P0 = 0.5 * (P0 + P0.T)
                return {"is_stabilizable": True, "K": np.zeros((0, n)), "P": P0}
            lam = eigvals(A)
            if np.max(np.abs(lam)) < eps:
                P0 = solve_discrete_lyapunov(A.T, In)
                P0 = 0.5 * (P0 + P0.T)
                return {"is_stabilizable": True, "K": np.zeros((0, n)), "P": P0}
            return {"is_stabilizable": False, "K": None, "P": None}

        # Very cheap sufficient stability test to avoid DARE when K=0 works.
        # If ||A|| < 1 for any induced norm, then spectral radius(A) < 1.
        # Use 1-norm and inf-norm (O(n^2)).
        n1 = float(np.linalg.norm(A, ord=1))
        ni = float(np.linalg.norm(A, ord=np.inf))
        if n1 < eps or ni < eps:
            # Stronger certificate: ||A||2 <= sqrt(||A||1||A||inf) < 1 => P=I works.
            if (n1 * ni) ** 0.5 < eps:
                return {"is_stabilizable": True, "K": np.zeros((m, n)), "P": In}

            K0 = np.zeros((m, n))
            P0 = solve_discrete_lyapunov(A.T, In)
            P0 = 0.5 * (P0 + P0.T)
            if np.isfinite(P0).all():
                return {"is_stabilizable": True, "K": K0, "P": P0}

        Q = In
        R = _eye_m(m)

        # DARE solve: for Q>0, a stabilizing solution exists iff (A,B) is stabilizable.
        try:
            P = solve_discrete_are(A, B, Q, R)
        except Exception:
            # If DARE fails, distinguish true non-stabilizability from rare numerics via PBH.
            if not _pbh_stabilizable(A, B):
                return {"is_stabilizable": False, "K": None, "P": None}
            try:
                P = solve_discrete_are(A, B, Q * 10.0, R)
            except Exception:
                return {"is_stabilizable": False, "K": None, "P": None}

        if not np.isfinite(P).all():
            return {"is_stabilizable": False, "K": None, "P": None}

        BtP = B.T @ P
        G = R + BtP @ B
        try:
            K = -np.linalg.solve(G, BtP @ A)
        except np.linalg.LinAlgError:
            return {"is_stabilizable": False, "K": None, "P": None}

        # Enforce symmetry numerically (validator checks symmetry).
        P = 0.5 * (P + P.T)

        if not (np.isfinite(P).all() and np.isfinite(K).all()):
            return {"is_stabilizable": False, "K": None, "P": None}

        return {"is_stabilizable": True, "K": K, "P": P}