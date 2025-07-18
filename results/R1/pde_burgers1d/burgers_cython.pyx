import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def burgers_equation_cython(double[::1] u, double nu, double inv_dx, double inv_dx2):
    cdef int n = u.shape[0]
    cdef double[::1] u_padded = np.zeros(n + 2)
    cdef double[::1] du_dt = np.zeros(n)
    cdef int i
    cdef double u_left, u_center, u_right, diffusion_term, du_dx, advection_term
    
    # Copy interior values
    for i in range(n):
        u_padded[i+1] = u[i]
    
    # Compute derivatives
    for i in range(1, n+1):
        u_left = u_padded[i-1]
        u_center = u_padded[i]
        u_right = u_padded[i+1]
        
        # Diffusion term
        diffusion_term = (u_right - 2*u_center + u_left) * inv_dx2
        
        # Advection term with upwind scheme
        if u_center >= 0:
            du_dx = (u_center - u_left) * inv_dx
        else:
            du_dx = (u_right - u_center) * inv_dx
            
        advection_term = u_center * du_dx
        du_dt[i-1] = -advection_term + nu * diffusion_term
        
    return np.asarray(du_dt)