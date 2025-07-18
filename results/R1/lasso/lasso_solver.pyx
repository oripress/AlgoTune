# distutils: language=c++
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange, parallel
from libc.math cimport fabs, fmax, fmin

@cython.boundscheck(False)
@cython.wraparound(False)
def coordinate_descent(cnp.ndarray[cnp.float64_t, ndim=2] X, 
                      cnp.ndarray[cnp.float64_t, ndim=1] y, 
                      double alpha=0.1, int max_iter=1000, double tol=1e-4):
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] w = np.zeros(d, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] residual = y.copy()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] diag = np.zeros(d, dtype=np.float64)
    
    cdef double grad_j, w_j_old, candidate, threshold, w_j_new, diff
    cdef int i, j, iter
    cdef double col_val
    
    # Precompute feature norms in parallel
    cdef double[:, :] X_view = X
    cdef double[:] diag_view = diag
    
    with nogil, parallel(num_threads=4):
        for j in prange(d, schedule='static'):
            diag_view[j] = 0.0
            for i in range(n):
                col_val = X_view[i, j]
                diag_view[j] += col_val * col_val
            diag_view[j] = fmax(diag_view[j], 1e-10)
    
    cdef double[:] residual_view = residual
    cdef double[:] w_view = w
    
    # Initialize residual norm squared and L1 norm
    cdef double r_norm_sq = 0.0
    for i in range(n):
        r_norm_sq += residual_view[i] * residual_view[i]
    cdef double l1_norm = 0.0
    cdef double best_obj = r_norm_sq / (2 * n) + alpha * l1_norm
    
    # Coordinate descent loop
    for iter in range(max_iter):
        cdef double max_change = 0.0
        
        # Use cyclic order for features
        for j in range(d):
            grad_j = 0.0
            
            # Calculate gradient using residual
            for i in range(n):
                grad_j += X_view[i, j] * residual_view[i]
            
            # Soft-thresholding
            w_j_old = w_view[j]
            candidate = w_j_old + grad_j / diag_view[j]
            threshold = alpha * n / diag_view[j]
            
            if candidate > threshold:
                w_j_new = candidate - threshold
            elif candidate < -threshold:
                w_j_new = candidate + threshold
            else:
                w_j_new = 0.0
                
            # Update coefficient if changed
            if w_j_new != w_j_old:
                diff = w_j_new - w_j_old
                w_view[j] = w_j_new
                
                # Update residual
                for i in range(n):
                    residual_view[i] -= diff * X_view[i, j]
                
                # Update residual norm squared incrementally
                r_norm_sq = r_norm_sq - 2 * diff * grad_j + diff * diff * diag_view[j]
                
                # Update L1 norm
                l1_norm += fabs(w_j_new) - fabs(w_j_old)
                
                if fabs(diff) > max_change:
                    max_change = fabs(diff)
        
        # Compute current objective
        cdef double current_obj = r_norm_sq / (2 * n) + alpha * l1_norm
        
        # Early stopping every 10 iterations
        if iter % 10 == 0:
            if best_obj - current_obj < tol * best_obj and max_change < tol:
                break
            best_obj = fmin(best_obj, current_obj)
        elif max_change < tol:
            break
    
    return w