from typing import Any
import numpy as np
import numba as nb

try:
    from fast_solve import solve_general
    HAS_FAST = True
except ImportError:
    HAS_FAST = False

# Numba JIT for 4x4 Cramer's rule
@nb.njit(cache=True)
def solve_4x4(A, b):
    """Solve 4x4 system using cofactor expansion."""
    # Compute 2x2 minors for first two rows
    m00 = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    m01 = A[0,0]*A[1,2] - A[0,2]*A[1,0]
    m02 = A[0,0]*A[1,3] - A[0,3]*A[1,0]
    m03 = A[0,1]*A[1,2] - A[0,2]*A[1,1]
    m04 = A[0,1]*A[1,3] - A[0,3]*A[1,1]
    m05 = A[0,2]*A[1,3] - A[0,3]*A[1,2]
    
    # Compute 2x2 minors for last two rows
    m10 = A[2,0]*A[3,1] - A[2,1]*A[3,0]
    m11 = A[2,0]*A[3,2] - A[2,2]*A[3,0]
    m12 = A[2,0]*A[3,3] - A[2,3]*A[3,0]
    m13 = A[2,1]*A[3,2] - A[2,2]*A[3,1]
    m14 = A[2,1]*A[3,3] - A[2,3]*A[3,1]
    m15 = A[2,2]*A[3,3] - A[2,3]*A[3,2]
    
    # 4x4 determinant
    det = m00*m15 - m01*m14 + m02*m13 + m03*m12 - m04*m11 + m05*m10
    
    # Cofactors for each column replacement
    # x0: replace column 0 with b
    c00 = b[0]*A[1,1] - A[0,1]*b[1]
    c01 = b[0]*A[1,2] - A[0,2]*b[1]
    c02 = b[0]*A[1,3] - A[0,3]*b[1]
    c03 = A[0,1]*A[1,2] - A[0,2]*A[1,1]  # same as m03
    c04 = A[0,1]*A[1,3] - A[0,3]*A[1,1]  # same as m04
    c05 = A[0,2]*A[1,3] - A[0,3]*A[1,2]  # same as m05
    
    c10 = b[2]*A[3,1] - A[2,1]*b[3]
    c11 = b[2]*A[3,2] - A[2,2]*b[3]
    c12 = b[2]*A[3,3] - A[2,3]*b[3]
    c13 = A[2,1]*A[3,2] - A[2,2]*A[3,1]  # same as m13
    c14 = A[2,1]*A[3,3] - A[2,3]*A[3,1]  # same as m14
    c15 = A[2,2]*A[3,3] - A[2,3]*A[3,2]  # same as m15
    
    x0 = (c00*c15 - c01*c14 + c02*c13 + c03*c12 - c04*c11 + c05*c10) / det
    
    # This is getting complex - let's just use numpy solve
    x = np.linalg.solve(A, b)
    return x

class Solver:
    def __init__(self):
        # Warm up numba
        A4 = np.eye(4)
        b4 = np.ones(4)
        solve_4x4(A4, b4)
    
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        pA = problem["A"]
        pb = problem["b"]
        n = len(pb)
        
        if n == 1:
            return [pb[0] / pA[0][0]]
        elif n == 2:
            a00, a01 = pA[0][0], pA[0][1]
            a10, a11 = pA[1][0], pA[1][1]
            det = a00 * a11 - a01 * a10
            return [(a11 * pb[0] - a01 * pb[1]) / det, 
                    (-a10 * pb[0] + a00 * pb[1]) / det]
        elif n == 3:
            a = pA
            b = pb
            det = (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1]) 
                  - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0]) 
                  + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))
            x0 = (b[0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1]) 
                  - a[0][1]*(b[1]*a[2][2]-a[1][2]*b[2]) 
                  + a[0][2]*(b[1]*a[2][1]-a[1][1]*b[2])) / det
            x1 = (a[0][0]*(b[1]*a[2][2]-a[1][2]*b[2]) 
                  - b[0]*(a[1][0]*a[2][2]-a[1][2]*a[2][0]) 
                  + a[0][2]*(a[1][0]*b[2]-b[1]*a[2][0])) / det
            x2 = (a[0][0]*(a[1][1]*b[2]-b[1]*a[2][1]) 
                  - a[0][1]*(a[1][0]*b[2]-b[1]*a[2][0]) 
                  + b[0]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])) / det
            return [x0, x1, x2]
        
        if HAS_FAST:
            return solve_general(pA, pb, n)
        
        A = np.array(pA, dtype=np.float64)
        b = np.array(pb, dtype=np.float64)
        return np.linalg.solve(A, b).tolist()