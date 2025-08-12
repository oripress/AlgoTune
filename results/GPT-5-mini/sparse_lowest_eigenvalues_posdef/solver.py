from typing import Any, List, Dict
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        """
        Compute the k smallest eigenvalues of a (sparse) positive semi-definite matrix.

        Strategy:
        - Convert the input matrix to CSR format if possible.
        - For tiny systems or when k is close to n, use dense np.linalg.eigvalsh.
        - Otherwise use scipy.sparse.linalg.eigsh to compute the k smallest eigenvalues.
        - On ARPACK non-convergence, try to use partial results; otherwise fall back to dense.
        """
        # Extract inputs
        mat_in = problem.get("matrix")
        if mat_in is None:
            return []

        try:
            k = int(problem.get("k", kwargs.get("k", 5)))
        except Exception:
            k = 5

        if k <= 0:
            return []

        # Convert input to CSR if possible
        try:
            if hasattr(mat_in, "asformat") and callable(getattr(mat_in, "asformat")):
                mat = mat_in.asformat("csr")
            elif sparse.isspmatrix(mat_in):
                mat = mat_in.tocsr()
            else:
                arr = np.asarray(mat_in)
                mat = sparse.csr_matrix(arr)
        except Exception:
            arr = np.asarray(mat_in)
            mat = sparse.csr_matrix(arr)

        # Validate shape
        try:
            n, m = mat.shape
        except Exception:
            # Fallback: treat as dense array
            arr = np.asarray(mat)
            vals = np.linalg.eigvalsh(arr)
            vals = np.sort(np.real(vals))
            return [float(v) for v in vals[:k]]

        if n == 0:
            return []

        # Dense path for tiny systems or when k is close to n (heuristic from reference)
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            vals = np.sort(np.real(vals))
            return [float(v) for v in vals[:k]]

        # Sparse ARPACK path (mirror reference parameters)
        try:
            ncv = min(n - 1, max(2 * k + 1, 20))
            vals = eigsh(
                mat,
                k=k,
                which="SM",  # smallest magnitude eigenvalues (for PSD, smallest algebraic)
                return_eigenvectors=False,
                maxiter=max(100, n * 200),
                ncv=ncv,
            )
            vals = np.sort(np.real(vals))
            return [float(v) for v in vals[:k]]
        except ArpackNoConvergence as e:
            # Try to use partial results if provided by ARPACK
            ev = getattr(e, "eigenvalues", None)
            if ev is not None and len(ev) >= k:
                try:
                    ev_sorted = np.sort(np.real(ev))
                    return [float(v) for v in ev_sorted[:k]]
                except Exception:
                    pass
            # Fall back to dense
            vals = np.linalg.eigvalsh(mat.toarray())
            vals = np.sort(np.real(vals))
            return [float(v) for v in vals[:k]]
        except Exception:
            # Generic fallback to dense
            vals = np.linalg.eigvalsh(mat.toarray())
            vals = np.sort(np.real(vals))
            return [float(v) for v in vals[:k]]