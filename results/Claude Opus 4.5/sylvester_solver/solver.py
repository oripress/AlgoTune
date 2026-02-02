from typing import Any
import numpy as np
from scipy.linalg import solve_sylvester, eig, lu_factor, lu_solve

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        A, B, Q = problem["A"], problem["B"], problem["Q"]
        
        try:
            # Eigendecomposition approach - faster for well-conditioned matrices
            DA, PA = eig(A, check_finite=False, overwrite_a=False)
            DB, PB = eig(B, check_finite=False, overwrite_a=False)
            
            # LU factorization is faster than general solve for multiple RHS
            lu_PA, piv_PA = lu_factor(PA, check_finite=False)
            lu_PB, piv_PB = lu_factor(PB.T, check_finite=False)
            
            # F = PA^{-1} @ Q @ PB
            QP = Q @ PB
            F = lu_solve((lu_PA, piv_PA), QP, check_finite=False)
            
            # Y[i,j] = F[i,j] / (DA[i] + DB[j])
            Y = F / np.add.outer(DA, DB)
            
            # X = PA @ Y @ PB^{-1}
            PY = PA @ Y
            X = lu_solve((lu_PB, piv_PB), PY.T, check_finite=False).T
            
            if np.isfinite(X).all():
                return {"X": X}
        except:
            pass
        
        # Fallback
        X = solve_sylvester(A, B, Q)
        return {"X": X}