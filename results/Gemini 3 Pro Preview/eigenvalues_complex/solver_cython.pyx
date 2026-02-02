import numpy as np
cimport numpy as np
from libc.math cimport sqrt

def solve_cython(double[:, ::1] problem):
    cdef int n = problem.shape[0]
    cdef int i, j
    cdef double a, b, c, d, trace, det, delta
    cdef double complex sqrt_delta, l1, l2
    cdef bint is_symmetric = True
    
    if n == 1:
        return [complex(problem[0, 0], 0.0)]
        
    if n == 2:
        a = problem[0, 0]
        b = problem[0, 1]
        c = problem[1, 0]
        d = problem[1, 1]
        
        trace = a + d
        det = a*d - b*c
        
        delta = trace*trace - 4*det
        sqrt_delta = (delta + 0j)**0.5
        
        l1 = (trace + sqrt_delta) / 2.0
        l2 = (trace - sqrt_delta) / 2.0
        
        if l1.real > l2.real:
            return [l1, l2]
        elif l2.real > l1.real:
            return [l2, l1]
        else:
            if l1.imag >= l2.imag:
                return [l1, l2]
            else:
                return [l2, l1]

    # Check symmetry manually
    with nogil:
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i, j] != problem[j, i]:
                    is_symmetric = False
                    break
            if not is_symmetric:
                break
                
    if is_symmetric:
        # Use numpy for eigvalsh as it's optimized LAPACK
        vals = np.linalg.eigvalsh(problem)
        return [complex(x, 0.0) for x in vals[::-1]]
    else:
        vals = np.linalg.eigvals(problem)
        vals.sort()
        return vals[::-1].tolist()