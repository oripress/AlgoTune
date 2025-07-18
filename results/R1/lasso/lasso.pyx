import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def coordinate_descent(double[:, ::1] X, double[::1] y, double alpha, double tol=1e-4, int max_iter=1000):
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef double[::1] w = np.zeros(n_features)
    cdef double[::1] r = np.array(y)
    
    # Precompute column norms
    cdef double[::1] norm_cols = np.zeros(n_features)
    cdef double col_norm
    cdef int i, j, it
    for j in range(n_features):
        col_norm = 0.0
        for i in range(n_samples):
            col_norm += X[i, j] * X[i, j]
        norm_cols[j] = col_norm
    
    cdef double w_j_old, grad, candidate, threshold, w_j_new, diff
    cdef double max_change, change
    cdef double prev_obj = np.inf
    cdef double obj
    
    for it in range(max_iter):
        max_change = 0.0
        for j in range(n_features):
            if norm_cols[j] < 1e-10:
                continue
                
            w_j_old = w[j]
            grad = 0.0
            for i in range(n_samples):
                grad += X[i, j] * r[i]
            
            candidate = w_j_old + grad / norm_cols[j]
            threshold = alpha / norm_cols[j]
            
            if candidate > threshold:
                w_j_new = candidate - threshold
            elif candidate < -threshold:
                w_j_new = candidate + threshold
            else:
                w_j_new = 0.0
                
            if w_j_new != w_j_old:
                diff = w_j_old - w_j_new
                for i in range(n_samples):
                    r[i] += diff * X[i, j]
                w[j] = w_j_new
                change = fabs(diff)
                if change > max_change:
                    max_change = change
        
        # Compute objective for convergence check
        obj = 0.0
        for i in range(n_samples):
            obj += r[i] * r[i]
        obj = 0.5 * obj / n_samples
        for j in range(n_features):
            obj += alpha * fabs(w[j])
            
        if fabs(prev_obj - obj) < tol:
            break
        prev_obj = obj
            
    return np.asarray(w)