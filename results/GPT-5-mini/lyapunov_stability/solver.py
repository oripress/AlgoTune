from typing import Any, Dict, Optional
import numpy as np

# Try to import SciPy's discrete Lyapunov solver if available
try:
    from scipy.linalg import solve_discrete_lyapunov as _solve_discrete_lyapunov
except Exception:
    _solve_discrete_lyapunov = None

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Determine discrete-time asymptotic stability by finding P > 0 such that:
            A.T @ P @ A - P < 0

        Returns:
            {"is_stable": bool, "P": n x n list or None}
        """
        # Validate input
        if not isinstance(problem, dict) or "A" not in problem:
            return {"is_stable": False, "P": None}
        try:
            A = np.array(problem["A"], dtype=float)
        except Exception:
            return {"is_stable": False, "P": None}
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return {"is_stable": False, "P": None}
        n = A.shape[0]
        if n == 0:
            return {"is_stable": False, "P": None}

        # Tolerances and parameters
        tol_eig = float(kwargs.get("tol_eig", 1e-10))
        imag_tol = float(kwargs.get("imag_tol", 1e-12))
        kronecker_max_n = int(kwargs.get("kronecker_max_n", 40))
        series_max_iters = int(kwargs.get("series_max_iters", 5000))
        series_tol = float(kwargs.get("series_tol", 1e-14))

        I = np.eye(n)

        # Quick necessary condition: spectral radius < 1
        try:
            eigs = np.linalg.eigvals(A)
        except Exception:
            return {"is_stable": False, "P": None}
        if not np.all(np.isfinite(eigs)):
            return {"is_stable": False, "P": None}
        rho = float(np.max(np.abs(eigs)))
        # If spectral radius >= 1, system is not asymptotically stable
        if rho >= 1.0 - 1e-15:
            return {"is_stable": False, "P": None}

        def _finalize_and_check(Pcand: Optional[np.ndarray]) -> Optional[np.ndarray]:
            """Symmetrize, remove tiny imag parts, ensure P positive definite and S negative definite."""
            if Pcand is None:
                return None

            # Remove tiny imaginary parts
            if np.iscomplexobj(Pcand):
                max_im = float(np.max(np.abs(Pcand.imag)))
                if not np.isfinite(max_im):
                    return None
                if max_im < imag_tol:
                    Pcand = Pcand.real
                else:
                    return None

            if not np.all(np.isfinite(Pcand)):
                return None

            # Symmetrize
            Pcand = 0.5 * (Pcand + Pcand.T)

            # Fast check for positive definiteness using Cholesky
            try:
                np.linalg.cholesky(Pcand)
                pd_ok = True
            except np.linalg.LinAlgError:
                pd_ok = False
            except Exception:
                return None

            # If Cholesky failed, try nudging tiny negative eigenvalues
            if not pd_ok:
                try:
                    eigP = np.linalg.eigvalsh(Pcand)
                except Exception:
                    return None
                if not np.all(np.isfinite(eigP)):
                    return None
                min_eig = float(np.min(eigP))
                if min_eig <= 0.0 and min_eig > -1e-12:
                    Pcand = Pcand + (abs(min_eig) + 1e-12) * I
                    try:
                        np.linalg.cholesky(Pcand)
                        pd_ok = True
                    except Exception:
                        return None
                else:
                    return None

            # At this point Pcand is positive definite; ensure minimum eigenvalue meets tol_eig
            try:
                eigP = np.linalg.eigvalsh(Pcand)
            except Exception:
                return None
            if not np.all(np.isfinite(eigP)):
                return None
            min_eig = float(np.min(eigP))
            if min_eig < tol_eig:
                if min_eig <= 0.0:
                    return None
                scale = tol_eig / min_eig
                if not np.isfinite(scale) or scale <= 0.0:
                    return None
                Pcand = Pcand * scale
                Pcand = 0.5 * (Pcand + Pcand.T)
                try:
                    np.linalg.cholesky(Pcand)
                except Exception:
                    return None

            # Check Lyapunov inequality S = A.T P A - P should be negative definite (allow small slack)
            try:
                S = A.T @ Pcand @ A - Pcand
            except Exception:
                return None

            if np.iscomplexobj(S):
                max_im = float(np.max(np.abs(S.imag)))
                if not np.isfinite(max_im):
                    return None
                if max_im < imag_tol:
                    S = S.real
                else:
                    return None

            # Fast check for negative definiteness using Cholesky on -S
            try:
                np.linalg.cholesky(-S)
            except np.linalg.LinAlgError:
                # Fall back to eigenvalue check if Cholesky fails (allow small positive tolerance)
                try:
                    eigS = np.linalg.eigvalsh(S)
                except Exception:
                    return None
                if not np.all(np.isfinite(eigS)):
                    return None
                if float(np.max(eigS)) > tol_eig:
                    return None
            except Exception:
                return None

            return Pcand

        # Strategy 1: SciPy's fast solver (if available)
        if _solve_discrete_lyapunov is not None:
            try:
                # scipy.linalg.solve_discrete_lyapunov(a, q) solves a X a^H - X + q = 0
                # We want A.T P A - P = -I => set a = A.T, q = I
                P_try = _solve_discrete_lyapunov(A.T, I)
                P_ok = _finalize_and_check(P_try)
                if P_ok is not None:
                    return {"is_stable": True, "P": P_ok.tolist()}
            except Exception:
                pass

        # Strategy 2: Kronecker linear solve for moderate n
        if n <= kronecker_max_n:
            try:
                N = n * n
                # vec(A.T P A) = (A.T ⊗ A.T) vec(P); solve (I - Kron) vec(P) = vec(I)
                M = np.eye(N) - np.kron(A.T, A.T)
                vecI = I.reshape(N, order="F")
                vecP = np.linalg.solve(M, vecI)
                P_try = vecP.reshape((n, n), order="F")
                P_ok = _finalize_and_check(P_try)
                if P_ok is not None:
                    return {"is_stable": True, "P": P_ok.tolist()}
            except Exception:
                pass

        # Strategy 3: Series summation fallback: P = sum_{k=0}^\infty (A.T)^k I A^k
        try:
            P_try = np.zeros((n, n), dtype=float)
            term = I.copy()
            P_try += term
            for _ in range(1, series_max_iters):
                term = A.T @ term @ A
                term_norm = np.linalg.norm(term, ord=2)
                if not np.isfinite(term_norm) or term_norm == 0.0:
                    break
                P_try += term
                if term_norm <= series_tol:
                    break
            P_ok = _finalize_and_check(P_try)
            if P_ok is not None:
                return {"is_stable": True, "P": P_ok.tolist()}
        except Exception:
            pass

        # If none succeeded, report unstable / undecidable
        return {"is_stable": False, "P": None}