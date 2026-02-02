import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def extract_results(double[::1, :] lu_packed, int[:] piv, int n):
    # lu_packed is F-contiguous (column-major)
    
    cdef int i, j, target
    cdef double val
    
    # Construct p vector
    cdef int[:] p = np.arange(n, dtype=np.int32)
    cdef int tmp
    for i in range(n):
        target = piv[i]
        if target != i:
            tmp = p[i]
            p[i] = p[target]
            p[target] = tmp
            
    # Construct inv_p for P matrix
    cdef int[:] inv_p = np.zeros(n, dtype=np.int32)
    for i in range(n):
        inv_p[p[i]] = i
        
    # Create lists
    L_list = []
    U_list = []
    P_list = []
    
    cdef list row_l, row_u, row_p
    
    for i in range(n):
        row_l = [0.0] * n
        row_u = [0.0] * n
        row_p = [0.0] * n
        
        # Fill P
        row_p[inv_p[i]] = 1.0
        P_list.append(row_p)
        
        # Fill L and U
        # L has 1 on diagonal, lower part from lu_packed
        # U has upper part from lu_packed
        
        # Diagonal
        row_l[i] = 1.0
        row_u[i] = lu_packed[i, i]
        
        # Lower part (j < i) -> L
        for j in range(i):
            row_l[j] = lu_packed[i, j]
            
        # Upper part (j > i) -> U
        for j in range(i + 1, n):
            row_u[j] = lu_packed[i, j]
            
        L_list.append(row_l)
        U_list.append(row_u)
        
    return {"P": P_list, "L": L_list, "U": U_list}