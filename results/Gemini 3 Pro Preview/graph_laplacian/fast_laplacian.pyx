# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt, fabs
from libc.stdlib cimport malloc, free, qsort

cdef struct Entry:
    int col
    double val

cdef int compare_entries(const void *a, const void *b) noexcept nogil:
    cdef Entry *entry_a = <Entry *>a
    cdef Entry *entry_b = <Entry *>b
    if entry_a.col < entry_b.col:
        return -1
    elif entry_a.col > entry_b.col:
        return 1
    else:
        return 0

cdef void process_row_degree(int i, double[:] data, int[:] indptr, double[:] degrees) noexcept nogil:
    cdef int start = indptr[i]
    cdef int end = indptr[i+1]
    cdef double d = 0.0
    cdef int j
    for j in range(start, end):
        d += data[j]
    degrees[i] = d

cdef void process_row_nnz(int i, double[:] data, int[:] indices, int[:] indptr, double[:] degrees, double[:] w, int[:] row_nnz, bint normed) noexcept nogil:
    cdef int start = indptr[i]
    cdef int end = indptr[i+1]
    cdef int cnt = 0
    cdef bint diag_seen = False
    cdef int j, col
    cdef double val, l_val
    
    if normed and fabs(degrees[i]) <= 1e-14:
        row_nnz[i] = 0
        return

    for j in range(start, end):
        col = indices[j]
        val = data[j]
        
        if col == i:
            diag_seen = True
            if normed:
                l_val = 1.0 - val * w[i] * w[i]
            else:
                l_val = degrees[i] - val
        else:
            if normed:
                l_val = -val * w[i] * w[col]
            else:
                l_val = -val
        
        if fabs(l_val) > 1e-14:
            cnt += 1
    
    if not diag_seen:
        if normed:
            cnt += 1
        else:
            if fabs(degrees[i]) > 1e-14:
                cnt += 1
    
    row_nnz[i] = cnt

cdef void process_row_fill(int i, double[:] data, int[:] indices, int[:] indptr, double[:] degrees, double[:] w, int[:] out_indptr, double[:] out_data, int[:] out_indices, bint normed) noexcept nogil:
    cdef int start = indptr[i]
    cdef int end = indptr[i+1]
    cdef int out_start = out_indptr[i]
    cdef int ptr = 0
    cdef bint diag_seen = False
    cdef int j, col
    cdef double val, l_val
    cdef Entry *row_buffer
    
    if normed and fabs(degrees[i]) <= 1e-14:
        return

    row_buffer = <Entry *> malloc((end - start + 1) * sizeof(Entry))
    if row_buffer == NULL:
        return

    for j in range(start, end):
        col = indices[j]
        val = data[j]
        
        if col == i:
            diag_seen = True
            if normed:
                l_val = 1.0 - val * w[i] * w[i]
            else:
                l_val = degrees[i] - val
        else:
            if normed:
                l_val = -val * w[i] * w[col]
            else:
                l_val = -val
        
        if fabs(l_val) > 1e-14:
            row_buffer[ptr].col = col
            row_buffer[ptr].val = l_val
            ptr += 1
    
    if not diag_seen:
        if normed:
            row_buffer[ptr].col = i
            row_buffer[ptr].val = 1.0
            ptr += 1
        else:
            if fabs(degrees[i]) > 1e-14:
                row_buffer[ptr].col = i
                row_buffer[ptr].val = degrees[i]
                ptr += 1
    
    qsort(row_buffer, ptr, sizeof(Entry), compare_entries)
    
    for j in range(ptr):
        out_indices[out_start + j] = row_buffer[j].col
        out_data[out_start + j] = row_buffer[j].val
        
    free(row_buffer)

def compute_laplacian_cython(double[:] data, int[:] indices, int[:] indptr, int n, bint normed):
    cdef double[:] degrees = np.zeros(n, dtype=np.float64)
    cdef double[:] w
    cdef int[:] row_nnz = np.zeros(n, dtype=np.int32)
    cdef int[:] out_indptr = np.empty(n + 1, dtype=np.int32)
    cdef int total_nnz
    cdef double[:] out_data
    cdef int[:] out_indices
    cdef int i

    # 1. Compute degrees
    for i in prange(n, nogil=True):
        process_row_degree(i, data, indptr, degrees)

    # Precompute weights
    if normed:
        w = np.zeros(n, dtype=np.float64)
        for i in prange(n, nogil=True):
            if fabs(degrees[i]) > 1e-14:
                w[i] = 1.0 / sqrt(degrees[i])
    else:
        # Dummy w to avoid uninitialized memory access if passed (though not used)
        # But we can pass None or handle inside.
        # Cython typed memoryviews cannot be None if typed.
        # So we allocate a dummy one or handle logic.
        w = np.zeros(1, dtype=np.float64) # Dummy
    
    # 2. Count NNZ
    for i in prange(n, nogil=True):
        process_row_nnz(i, data, indices, indptr, degrees, w, row_nnz, normed)

    # 3. Prefix sum (Sequential)
    out_indptr[0] = 0
    total_nnz = 0
    for i in range(n):
        total_nnz += row_nnz[i]
        out_indptr[i+1] = total_nnz
        
    out_data = np.empty(total_nnz, dtype=np.float64)
    out_indices = np.empty(total_nnz, dtype=np.int32)
    
    # 4. Fill and Sort
    for i in prange(n, nogil=True):
        process_row_fill(i, data, indices, indptr, degrees, w, out_indptr, out_data, out_indices, normed)

    return np.asarray(out_data), np.asarray(out_indices), np.asarray(out_indptr)