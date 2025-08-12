import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Compute the principal matrix square root X such that X @ X = A and the
        eigenvalues of X have non-negative real parts.

        Strategy:
          - Parse input into a complex numpy array (handles strings like "1+2j").
          - Fast paths:
              * 1x1 scalar
              * Hermitian (use eigh)
              * General diagonalizable via eig + linear solve (avoids explicit inverse)
          - Fallback: scipy.linalg.sqrtm for robustness.

        Returns:
            {"sqrtm": {"X": nested_list_of_complex}} or {"sqrtm": {"X": []}} on failure.
        """
        # Tolerances for verification
        rtol = 1e-5
        atol = 1e-8

        # Extract input
        try:
            A_raw = problem["matrix"]
        except Exception:
            return {"sqrtm": {"X": []}}

        # Convert to a complex numpy array robustly
        try:
            A = np.array(A_raw, dtype=complex)
        except Exception:
            try:
                A = np.array([[complex(x) for x in row] for row in A_raw], dtype=complex)
            except Exception:
                return {"sqrtm": {"X": []}}

        # Validate shape: must be square matrix
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return {"sqrtm": {"X": []}}

        n = A.shape[0]

        # Trivial cases
        if n == 0:
            return {"sqrtm": {"X": []}}
        if n == 1:
            try:
                val = complex(np.sqrt(A[0, 0]))
                return {"sqrtm": {"X": [[val]]}}
            except Exception:
                return {"sqrtm": {"X": []}}

        def _pack(X: np.ndarray) -> Dict[str, Dict[str, Any]]:
            """Convert numpy array X to nested Python lists (complex numbers allowed)."""
            try:
                X = np.array(X, dtype=complex)
                return {"sqrtm": {"X": X.tolist()}}
            except Exception:
                try:
                    return {"sqrtm": {"X": [[complex(el) for el in row] for row in X]}}
                except Exception:
                    return {"sqrtm": {"X": []}}

        # 1) Hermitian (or real symmetric) fast path using eigh
        try:
            if np.allclose(A, A.conj().T, rtol=rtol, atol=atol):
                w, u = np.linalg.eigh(A)
                # Use complex dtype before sqrt to avoid issues with negative eigenvalues
                sqrt_w = np.sqrt(w.astype(complex))
                X = (u * sqrt_w[np.newaxis, :]) @ u.conj().T
                if np.all(np.isfinite(X)) and np.allclose(X @ X, A, rtol=rtol, atol=atol):
                    return _pack(X)
        except Exception:
            pass

        # 2) General eigen-decomposition approach for diagonalizable matrices
        try:
            w, v = np.linalg.eig(A)
            sqrt_w = np.sqrt(w.astype(complex))
            M = v * sqrt_w[np.newaxis, :]
            # Solve v.T * X.T = M.T  =>  X = solve(v.T, M.T).T (avoids explicit inverse)
            X = np.linalg.solve(v.T, M.T).T
            if np.all(np.isfinite(X)) and np.allclose(X @ X, A, rtol=rtol, atol=atol):
                return _pack(X)
        except Exception:
            pass

        # 3) Robust fallback to scipy.linalg.sqrtm
        try:
            import scipy.linalg as la

            try:
                out = la.sqrtm(A, disp=False)
            except TypeError:
                # Older/newer scipy may not accept disp argument
                out = la.sqrtm(A)
            X = out[0] if isinstance(out, tuple) else out
            X = np.array(X, dtype=complex)
            if not np.all(np.isfinite(X)):
                return {"sqrtm": {"X": []}}
            # Accept small numerical errors; reject gross failures
            if not np.allclose(X @ X, A, rtol=rtol, atol=atol):
                if np.max(np.abs(X @ X - A)) > 1e-6:
                    return {"sqrtm": {"X": []}}
            return _pack(X)
        except Exception:
            return {"sqrtm": {"X": []}}