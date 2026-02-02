import numpy as np
cimport numpy as np
cimport cython

np.import_array()

ctypedef fused DTYPE_t:
    np.float64_t
    np.complex128_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _upfirdn_impl(DTYPE_t[:] h, DTYPE_t[:] x, int up, int down, DTYPE_t[:] y) nogil:
    cdef int len_x = x.shape[0]
    cdef int len_h = h.shape[0]
    cdef int len_y = y.shape[0]
    cdef int n, t, max_j, min_j, j, val, h_idx
    cdef DTYPE_t acc
    
    for n in range(len_y):
        t = n * down
        
        max_j = t // up
        if max_j >= len_x:
            max_j = len_x - 1
            
        val = t - len_h + 1
        if val <= 0:
            min_j = 0
        else:
            min_j = (val + up - 1) // up
            
        acc = 0
        h_idx = t - min_j * up
        
        for j in range(min_j, max_j + 1):
            acc = acc + x[j] * h[h_idx]
            h_idx = h_idx - up
            
        y[n] = acc

def solve_cython(list problem):
    cdef list results = []
    cdef object h_obj, x_obj
    cdef int up, down
    cdef np.ndarray h_arr, x_arr, y_arr
    cdef int len_x, len_h, L_full, len_y
    cdef object dtype
    
    for item in problem:
        h_obj = item[0]
        x_obj = item[1]
        up = item[2]
        down = item[3]
        
        # Convert to array
        h_arr = np.asarray(h_obj)
        x_arr = np.asarray(x_obj)
        
        if x_arr.size == 0:
            dt = np.result_type(h_arr.dtype, x_arr.dtype)
            results.append(np.array([], dtype=dt))
            continue
            
        # Determine common type
        if np.iscomplexobj(h_arr) or np.iscomplexobj(x_arr):
            dtype = np.complex128
        else:
            dtype = np.float64
            
        # Cast if necessary
        if h_arr.dtype != dtype:
            h_arr = h_arr.astype(dtype, copy=False)
        if x_arr.dtype != dtype:
            x_arr = x_arr.astype(dtype, copy=False)
            
        len_x = x_arr.shape[0]
        len_h = h_arr.shape[0]
        L_full = (len_x - 1) * up + len_h
        len_y = (L_full - 1) // down + 1
        
        y_arr = np.zeros(len_y, dtype=dtype)
        
        if dtype == np.float64:
            _upfirdn_impl[np.float64_t](h_arr, x_arr, up, down, y_arr)
        else:
            _upfirdn_impl[np.complex128_t](h_arr, x_arr, up, down, y_arr)
            
        results.append(y_arr)
        
    return results