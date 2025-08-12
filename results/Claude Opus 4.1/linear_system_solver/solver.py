import numpy as np
import scipy.linalg
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        """Solve the linear system Ax = b with optimization for small matrices."""
        A = problem["A"]
        b = problem["b"]
        n = len(b)
        
        # For 2x2 matrices, use direct formula
        if n == 2:
            a11, a12 = A[0][0], A[0][1]
            a21, a22 = A[1][0], A[1][1]
            b1, b2 = b[0], b[1]
            det = a11 * a22 - a12 * a21
            x1 = (a22 * b1 - a12 * b2) / det
            x2 = (a11 * b2 - a21 * b1) / det
            return [x1, x2]
        
        # For 3x3 matrices, use direct inverse formula
        elif n == 3:
            # Extract matrix elements
            a11, a12, a13 = A[0][0], A[0][1], A[0][2]
            a21, a22, a23 = A[1][0], A[1][1], A[1][2]
            a31, a32, a33 = A[2][0], A[2][1], A[2][2]
            b1, b2, b3 = b[0], b[1], b[2]
            
            # Calculate determinant
            det = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)
            
            # Calculate cofactors and solve directly
            x1 = ((a22*a33 - a23*a32)*b1 - (a12*a33 - a13*a32)*b2 + (a12*a23 - a13*a22)*b3) / det
            x2 = (-(a21*a33 - a23*a31)*b1 + (a11*a33 - a13*a31)*b2 - (a11*a23 - a13*a21)*b3) / det
            x3 = ((a21*a32 - a22*a31)*b1 - (a11*a32 - a12*a31)*b2 + (a11*a22 - a12*a21)*b3) / det
            
            return [x1, x2, x3]
        
        # For larger matrices, use scipy's LU decomposition which is often faster
        else:
            # Use float32 for smaller matrices to reduce memory bandwidth
            if n <= 10:
                A_np = np.array(A, dtype=np.float32)
                b_np = np.array(b, dtype=np.float32)
                lu, piv = scipy.linalg.lu_factor(A_np, check_finite=False)
                x = scipy.linalg.lu_solve((lu, piv), b_np, check_finite=False)
                return x.astype(np.float64).tolist()
            else:
                A_np = np.array(A, dtype=np.float64)
                b_np = np.array(b, dtype=np.float64)
                lu, piv = scipy.linalg.lu_factor(A_np, check_finite=False)
                x = scipy.linalg.lu_solve((lu, piv), b_np, check_finite=False)
                return x.tolist()