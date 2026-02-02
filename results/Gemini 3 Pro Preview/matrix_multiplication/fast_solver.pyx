import numpy as np
cimport numpy as np
cimport cython
from cpython.list cimport PyList_GET_ITEM, PyList_GET_SIZE
from cpython.object cimport PyObject

@cython.boundscheck(False)
@cython.wraparound(False)
def solve_cython(object A_obj, object B_obj):
    cdef list A_list
    cdef list B_list
    
    if isinstance(A_obj, list):
        A_list = A_obj
    else:
        return np.dot(A_obj, B_obj)

    if isinstance(B_obj, list):
        B_list = B_obj
    else:
        return np.dot(A_obj, B_obj)

    cdef int n = PyList_GET_SIZE(A_list)
    if n == 0:
        return np.array([])
    
    # Assume all rows have same length, check first row
    cdef list first_row = <list>PyList_GET_ITEM(A_list, 0)
    cdef int m = PyList_GET_SIZE(first_row)
    
    if PyList_GET_SIZE(B_list) == 0:
        return np.zeros((n, 0))
    
    cdef list first_row_b = <list>PyList_GET_ITEM(B_list, 0)
    cdef int p = PyList_GET_SIZE(first_row_b)

    cdef np.ndarray[np.float64_t, ndim=2] A = np.empty((n, m), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] B = np.empty((m, p), dtype=np.float64)
    
    cdef int i, j
    cdef list row
    cdef object val_obj
    cdef double val

    for i in range(n):
        row = <list>PyList_GET_ITEM(A_list, i)
        for j in range(m):
            val_obj = <object>PyList_GET_ITEM(row, j)
            A[i, j] = val_obj

    for i in range(m):
        row = <list>PyList_GET_ITEM(B_list, i)
        for j in range(p):
            val_obj = <object>PyList_GET_ITEM(row, j)
            B[i, j] = val_obj
            
    return np.dot(A, B)
def convert_to_arrays(object A_obj, object B_obj):
    cdef list A_list
    cdef list B_list
    
    if isinstance(A_obj, list):
        A_list = A_obj
    else:
        # If already array, just return it
        return A_obj, B_obj

    if isinstance(B_obj, list):
        B_list = B_obj
    else:
        return A_obj, B_obj

    cdef int n = PyList_GET_SIZE(A_list)
    if n == 0:
        return np.array([]), np.array([])
    
    cdef list first_row = <list>PyList_GET_ITEM(A_list, 0)
    cdef int m = PyList_GET_SIZE(first_row)
    
    if PyList_GET_SIZE(B_list) == 0:
        return np.zeros((n, 0)), np.zeros((0, 0)) # Dimensions might be tricky here
    
    cdef list first_row_b = <list>PyList_GET_ITEM(B_list, 0)
    cdef int p = PyList_GET_SIZE(first_row_b)

    cdef np.ndarray[np.float64_t, ndim=2] A = np.empty((n, m), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] B = np.empty((m, p), dtype=np.float64)
    
    cdef int i, j
    cdef list row
    cdef object val_obj

    for i in range(n):
        row = <list>PyList_GET_ITEM(A_list, i)
        for j in range(m):
            val_obj = <object>PyList_GET_ITEM(row, j)
            A[i, j] = val_obj

    for i in range(m):
        row = <list>PyList_GET_ITEM(B_list, i)
        for j in range(p):
            val_obj = <object>PyList_GET_ITEM(row, j)
            B[i, j] = val_obj
            
    return A, B
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solve_cython_manual(object A_obj, object B_obj):
    cdef list A_list
    cdef list B_list
    
    if isinstance(A_obj, list):
        A_list = A_obj
    else:
        return np.dot(A_obj, B_obj)

    if isinstance(B_obj, list):
        B_list = B_obj
    else:
        return np.dot(A_obj, B_obj)

    cdef int n = PyList_GET_SIZE(A_list)
    if n == 0:
        return np.array([])
    
    cdef list first_row = <list>PyList_GET_ITEM(A_list, 0)
    cdef int m = PyList_GET_SIZE(first_row)
    
    if PyList_GET_SIZE(B_list) == 0:
        return np.zeros((n, 0))
    
    cdef list first_row_b = <list>PyList_GET_ITEM(B_list, 0)
    cdef int p = PyList_GET_SIZE(first_row_b)

    cdef np.ndarray[np.float64_t, ndim=2] C = np.zeros((n, p), dtype=np.float64)
    cdef double[:, :] C_view = C
    
    # We need to access A and B elements. 
    # Converting to array first is O(N*M + M*P).
    # Then multiplication is O(N*M*P).
    # If we access list elements directly in the loop, it's very slow due to boxing/unboxing.
    # So converting to C array (or numpy array) is necessary.
    
    # Let's use the fast conversion we already have, but maybe into a C pointer directly?
    # Using numpy array is fine.
    
    cdef np.ndarray[np.float64_t, ndim=2] A = np.empty((n, m), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] B = np.empty((m, p), dtype=np.float64)
    
    cdef int i, j, k
    cdef list row
    cdef object val_obj
    
    for i in range(n):
        row = <list>PyList_GET_ITEM(A_list, i)
        for j in range(m):
            val_obj = <object>PyList_GET_ITEM(row, j)
            A[i, j] = val_obj

    for i in range(m):
        row = <list>PyList_GET_ITEM(B_list, i)
        for j in range(p):
            val_obj = <object>PyList_GET_ITEM(row, j)
            B[i, j] = val_obj
            
    # Now manual multiplication
    cdef double[:, :] A_view = A
    cdef double[:, :] B_view = B
    cdef double temp
    
    for i in range(n):
        for k in range(m):
            temp = A_view[i, k]
            for j in range(p):
                C_view[i, j] += temp * B_view[k, j]
                
    return C