from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    # SciPy provides a fast, robust Schur-based solver (O(n^3))
    from scipy.linalg import solve_discrete_lyapunov as _sdlyap

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _sdlyap = None  # type: ignore
    _HAVE_SCIPY = False

class Solver:
    def _compute_p_discrete_lyap(self, A: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute P solving A^T P A - P = -I using fast methods.
        Returns None if computation fails.
        """
        n = A.shape[0]
        I = np.eye(n, dtype=A.dtype)

        # Try SciPy's solver (two conventions exist depending on version/expectation).
        if _HAVE_SCIPY:
            try:
                # SciPy solves A P A^T - P = -Q; use A.T to match A^T P A - P = -I
                P = _sdlyap(A.T, I)
                # Symmetrize to remove minor asymmetries
                return (P + P.T) * 0.5
            except Exception:
                pass

        # Last resort: geometric series summation (may be slow if rho(A) ~ 1)
        try:
            # Quick spectral radius check to avoid divergence
            rho = max(abs(np.linalg.eigvals(A)))
            if rho >= 1.0:
                return None
            # P = sum_{k=0..∞} (A^T)^k I A^k, truncate when terms small
            P = np.eye(n, dtype=A.dtype)
            term = P.copy()
            # Stop when new term is negligible
            tol = 1e-12
            max_iter = 10000
            for _ in range(max_iter):
                term = A.T @ term @ A
                P_next = P + term
                if np.linalg.norm(term, ord="fro") <= tol * (1.0 + np.linalg.norm(P_next, ord="fro")):
                    P = P_next
                    break
                P = P_next
            P = (P + P.T) * 0.5
            return P
        except Exception:
            return None

    @staticmethod
    def _scale_P_for_numerics(P: np.ndarray, min_eig_target: float = 1e-6) -> np.ndarray:
        """
        Scale P by a positive scalar s to ensure its minimum eigenvalue is >= min_eig_target.
        Scaling preserves Lyapunov inequality sign since:
            A^T (s P) A - (s P) = s (A^T P A - P)
        """
        # Ensure symmetry first
        P = (P + P.T) * 0.5
        try:
            evals = np.linalg.eigvalsh(P)
        except np.linalg.LinAlgError:
            # Fall back to general eigvals if symmetric solver fails
            evals = np.linalg.eigvals(P).real
        min_eig = float(np.min(evals))
        if not np.isfinite(min_eig):
            return P
        if min_eig <= 0:
            # If not PD due to numerical noise, shift by tiny identity before scaling
            shift = max(1e-12 - min_eig, 0.0)
            if shift > 0:
                P = P + shift * np.eye(P.shape[0], dtype=P.dtype)
                min_eig = min_eig + shift
                min_eig = max(min_eig, 1e-16)
        if min_eig < min_eig_target:
            s = min_eig_target / max(min_eig, 1e-16)
            P = P * s
        return (P + P.T) * 0.5

    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        Fast Lyapunov stability analysis for discrete-time LTI systems.

        Given x[k+1] = A x[k], determine stability by finding P > 0 such that:
            A^T P A - P < 0
        We construct P as the solution of the discrete Lyapunov equation:
            A^T P A - P = -I
        which exists iff spectral radius(A) < 1.

        Returns:
            {
              "is_stable": bool,
              "P": 2D list or None
            }
        """
        try:
            A_in = problem.get("A", None)
            A = np.array(A_in, dtype=float)
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                return {"is_stable": False, "P": None}

            # Compute Lyapunov matrix via fast solver
            P = self._compute_p_discrete_lyap(A)

            if P is None or not np.all(np.isfinite(P)):
                # Unable to produce a valid P -> consider unstable
                return {"is_stable": False, "P": None}

            # Symmetrize and ensure positive definiteness
            P = (P + P.T) * 0.5
            try:
                evals = np.linalg.eigvalsh(P)
            except np.linalg.LinAlgError:
                evals = np.linalg.eigvals(P).real
            min_eig = float(np.min(evals))
            if not np.isfinite(min_eig) or min_eig <= 0.0:
                return {"is_stable": False, "P": None}
            if min_eig < 1e-10:
                P = P * (1e-10 / max(min_eig, 1e-300))

            return {"is_stable": True, "P": P.tolist()}

        except Exception:
            # Any unexpected issue -> conservative return
            return {"is_stable": False, "P": None}