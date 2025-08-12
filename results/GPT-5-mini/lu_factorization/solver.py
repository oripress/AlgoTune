import numpy as np
from typing import Any, Dict

# Try to import scipy.linalg.lu (high-level) and LAPACK getrf (lower-level) as a fallback
_scilu = None
_getrf = None
try:
    from scipy.linalg import lu as _scilu
except Exception:
    _scilu = None

try:
    from scipy.linalg.lapack import get_lapack_funcs
    _getrf = get_lapack_funcs(("getrf",), (np.array([0.0], dtype=np.float64),))[0]
except Exception:
    _getrf = None

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """
        Compute LU factorization A = P @ L @ U with partial (row) pivoting.

        Input:
            problem: {"matrix": [[...], ...]} (square matrix)

        Output:
            {"LU": {"P": P_list, "L": L_list, "U": U_list}}
        """
        if not isinstance(problem, dict):
            raise ValueError("Problem must be a dict containing 'matrix'.")
        A_in = problem.get("matrix")
        if A_in is None:
            raise ValueError("Problem must contain 'matrix' key.")

        # Convert to NumPy array of floats (float64)
        A = np.array(A_in, dtype=np.float64, copy=True)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Input matrix must be square.")
        n = A.shape[0]

        # Handle empty matrix
        if n == 0:
            return {"LU": {"P": [], "L": [], "U": []}}

        # Prefer scipy.linalg.lu when available; it returns P,L,U such that A = P @ L @ U
        if _scilu is not None:
            try:
                P, L, U = _scilu(A)
                return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}
            except Exception:
                # If high-level lu fails for some reason, fall back to lower-level routines
                pass

        # Fast path: use LAPACK getrf if available
        if _getrf is not None:
            # For LAPACK prefer Fortran-contiguous arrays to avoid internal copies
            Af = A if A.flags.f_contiguous else np.asfortranarray(A)
            # Call getrf; typical signature returns (lu, piv, info)
            lu, piv, info = _getrf(Af, overwrite_a=1)
            if info < 0:
                raise ValueError(f"Illegal value in getrf argument {-info}")
            # Construct L and U from combined LU matrix
            U = np.triu(lu)
            L = np.tril(lu, k=-1)
            np.fill_diagonal(L, 1.0)

            # piv is 1-based pivot indices from LAPACK indicating the row swapped with i-th row
            piv = np.asarray(piv, dtype=np.intc) - 1
            # Build permutation matrix P by applying the sequence of row swaps to the identity
            P = np.eye(n, dtype=float)
            for i in range(piv.size):
                j = int(piv[i])
                if i != j:
                    P[[i, j], :] = P[[j, i], :]
            return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}

        # Fallback: pure NumPy Gaussian elimination with partial pivoting
        U = A.copy()
        L = np.zeros((n, n), dtype=float)
        P_internal = np.eye(n, dtype=float)

        # Tolerance for detecting zero pivot
        normA = np.linalg.norm(A, ord=np.inf)
        eps = np.finfo(float).eps * max(1.0, normA) * n

        for k in range(n):
            # Find pivot row (index of max abs in column k from rows k..n-1)
            rel_idx = int(np.argmax(np.abs(U[k:, k])))
            piv = k + rel_idx

            # If pivot is effectively zero, set L diagonal and continue
            if abs(U[piv, k]) <= eps:
                L[k, k] = 1.0
                continue

            # Swap rows in U and record permutation; also swap computed L entries
            if piv != k:
                U[[k, piv], :] = U[[piv, k], :]
                P_internal[[k, piv], :] = P_internal[[piv, k], :]
                if k > 0:
                    L[[k, piv], :k] = L[[piv, k], :k]

            pivot = U[k, k]

            # Compute multipliers and update trailing submatrix
            if k + 1 < n:
                mult = U[k + 1 :, k] / pivot
                L[k + 1 :, k] = mult
                U[k + 1 :, k:] -= mult[:, None] * U[k, k:]

            # Set diagonal of L
            L[k, k] = 1.0

        # We maintained P_internal so that P_internal @ A = L @ U.
        # To return P satisfying A = P @ L @ U, use P_internal.T
        P_return = P_internal.T

        return {"LU": {"P": P_return.tolist(), "L": L.tolist(), "U": U.tolist()}}