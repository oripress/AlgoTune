from typing import Any  
import numpy as np  
from scipy.sparse.linalg import eigsh  
  
class Solver:  
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:  
        # Obtain CSR-format sparse matrix  
        mat = problem["matrix"].asformat("csr")  
  
        k = int(problem["k"])  
        n = mat.shape[0]  
  
        # Dense fallback for trivial or very small problems  
        if k >= n or n < 2 * k + 1:  
            arr = mat.toarray()  
            vals = np.linalg.eigvalsh(arr)  
            return [float(v) for v in vals[:k]]  
  
        # Prepare ARPACK parameters  
        ncv = min(n - 1, max(2 * k + 1, 20))  
        tol = 1e-6  
  
        # Compute smallest eigenvalues via ARPACK  
        try:  
            vals = eigsh(  
                mat,  
                k=k,  
                which="SM",  # smallest magnitude  
                return_eigenvectors=False,  
                ncv=ncv,  
                tol=tol,  
            )  
        except Exception:  
            # Fallback to dense if ARPACK fails  
            arr = mat.toarray()  
            vals = np.linalg.eigvalsh(arr)[:k]  
  
        # Return sorted real parts  
        vals = np.real(vals)  
        vals.sort()  
        return [float(v) for v in vals]