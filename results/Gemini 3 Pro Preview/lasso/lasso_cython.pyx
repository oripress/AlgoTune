import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport fabs
from scipy.linalg.cython_blas cimport sdot, saxpy, snrm2
# from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solve_lasso(float[::1, :] X, float[:] y, float alpha, int max_iter=1000, float tol=1e-4):
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef float[:] w = np.zeros(d, dtype=np.float32)
    cdef float[:] r = y.copy()
    cdef float[:] L = np.zeros(d, dtype=np.float32)
    cdef int j, it
    cdef float dot_prod, old_w, new_w, dw, max_dw, rho, neg_dw
    cdef float threshold = n * alpha
    cdef int one = 1
    
    # Precompute column norms squared
    for j in range(d):
        L[j] = snrm2(&n, &X[0, j], &one) ** 2
        
    for it in range(max_iter):
        max_dw = 0
        for j in range(d):
            if L[j] == 0:
                continue
                
            # Calculate X_j^T r
            dot_prod = sdot(&n, &X[0, j], &one, &r[0], &one)
            
            rho = dot_prod + L[j] * w[j]
            
            old_w = w[j]
            if rho > threshold:
                new_w = (rho - threshold) / L[j]
            elif rho < -threshold:
                new_w = (rho + threshold) / L[j]
            else:
                new_w = 0.0
            
            if new_w == old_w:
                continue

            w[j] = new_w
            
            dw = new_w - old_w
            if fabs(dw) > 1e-7:
                neg_dw = -dw
                saxpy(&n, &neg_dw, &X[0, j], &one, &r[0], &one)
                if fabs(dw) > max_dw:
                    max_dw = fabs(dw)
        
        if max_dw < tol:
            break
            
    return np.asarray(w)