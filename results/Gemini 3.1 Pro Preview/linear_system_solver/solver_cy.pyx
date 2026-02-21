import numpy as np
cimport numpy as np
from scipy.linalg.cython_lapack cimport dgesv

def solve_cy(object A_obj, object b_obj):
    cdef np.ndarray[np.float64_t, ndim=2, mode="fortran"] A = np.asarray(A_obj, dtype=np.float64, order='F')
    cdef np.ndarray[np.float64_t, ndim=1, mode="fortran"] b = np.asarray(b_obj, dtype=np.float64, order='F')
    
    if not A.flags.f_contiguous or not A.flags.writeable:
        A = A.copy(order='F')
    if not b.flags.f_contiguous or not b.flags.writeable:
        b = b.copy(order='F')
        
    cdef int n = A.shape[0]
    cdef np.ndarray[int, ndim=1, mode="c"] ipiv = np.empty(n, dtype=np.intc)
    cdef int info = 0
    cdef int nrhs = 1
    
    dgesv(&n, &nrhs, &A[0,0], &n, &ipiv[0], &b[0], &n, &info)
    
    return b.tolist()