import numpy as np
from scipy.linalg.lapack import dsygvd
import math

class Solver:
    def solve(self, problem, **kwargs):
        A, B = problem
        n = A.shape[0]
        
        if n == 0:
            return []
        if n == 1:
            return [float(A[0, 0] / B[0, 0])]
        
        if n == 2:
            # Analytical solution for 2x2
            # Cholesky of B: L = [[l11, 0], [l21, l22]]
            b00, b01, b11 = B[0, 0], B[0, 1], B[1, 1]
            l11 = math.sqrt(b00)
            l21 = b01 / l11
            l22 = math.sqrt(b11 - l21 * l21)
            
            # Linv
            li11 = 1.0 / l11
            li21 = -l21 / (l11 * l22)
            li22 = 1.0 / l22
            
            a00, a01, a11 = A[0, 0], A[0, 1], A[1, 1]
            
            # Atilde = Linv @ A @ Linv.T
            # temp = Linv @ A
            t00 = li11 * a00
            t01 = li11 * a01
            t10 = li21 * a00 + li22 * a01
            t11 = li21 * a01 + li22 * a11
            
            # Atilde = temp @ Linv.T
            at00 = t00 * li11 + t01 * li21
            at01 = t01 * li22
            at11 = t10 * li21 + t11 * li22
            
            # Eigenvalues of 2x2 symmetric matrix
            trace = at00 + at11
            det = at00 * at11 - at01 * at01
            disc = math.sqrt(max(trace * trace - 4.0 * det, 0.0))
            e1 = (trace + disc) / 2.0
            e2 = (trace - disc) / 2.0
            return [e1, e2]

        # Direct LAPACK dsygvd call (divide and conquer - faster for larger matrices)
        if A.flags['F_CONTIGUOUS'] and A.dtype == np.float64:
            A_f = A.copy(order='F')
        else:
            A_f = np.asfortranarray(A, dtype=np.float64)
        if B.flags['F_CONTIGUOUS'] and B.dtype == np.float64:
            B_f = B.copy(order='F')
        else:
            B_f = np.asfortranarray(B, dtype=np.float64)
        
        w, _, info = dsygvd(A_f, B_f, itype=1, jobz='N', uplo='L',
                            overwrite_a=1, overwrite_b=1)
        
        return w[::-1].tolist()