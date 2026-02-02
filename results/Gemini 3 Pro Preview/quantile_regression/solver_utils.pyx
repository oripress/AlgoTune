import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def prepare_model_data(double[:, ::1] X, double[:] y, double quantile, bint fit_intercept):
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int m_constraints = n_features + (1 if fit_intercept else 0)
    
    # a_values
    cdef np.ndarray[np.float64_t, ndim=1] a_values
    cdef double[:] a_values_view
    cdef int i, j, idx
    
    if fit_intercept:
        a_values = np.empty(n_samples * m_constraints, dtype=np.float64)
        a_values_view = a_values
        idx = 0
        for i in range(n_samples):
            for j in range(n_features):
                a_values_view[idx] = X[i, j]
                idx += 1
            a_values_view[idx] = 1.0
            idx += 1
    else:
        # If no intercept, a_values is just X raveled
        # We can return a copy to be safe
        a_values = np.array(X).ravel()
    
    # a_indices
    cdef np.ndarray[np.int32_t, ndim=1] a_indices = np.empty(n_samples * m_constraints, dtype=np.int32)
    cdef int[:] a_indices_view = a_indices
    idx = 0
    cdef int k
    for i in range(n_samples):
        for k in range(m_constraints):
            a_indices_view[idx] = k
            idx += 1
            
    # a_start
    cdef np.ndarray[np.int32_t, ndim=1] a_start = np.empty(n_samples + 1, dtype=np.int32)
    cdef int[:] a_start_view = a_start
    for i in range(n_samples + 1):
        a_start_view[i] = i * m_constraints
        
    # col_cost
    cdef np.ndarray[np.float64_t, ndim=1] col_cost = np.empty(n_samples, dtype=np.float64)
    cdef double[:] col_cost_view = col_cost
    for i in range(n_samples):
        col_cost_view[i] = -y[i]
        
    # bounds
    cdef np.ndarray[np.float64_t, ndim=1] col_lower = np.empty(n_samples, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] col_upper = np.empty(n_samples, dtype=np.float64)
    cdef double[:] col_lower_view = col_lower
    cdef double[:] col_upper_view = col_upper
    cdef double q_minus_1 = quantile - 1.0
    
    for i in range(n_samples):
        col_lower_view[i] = q_minus_1
        col_upper_view[i] = quantile
        
    return a_values, a_indices, a_start, col_cost, col_lower, col_upper