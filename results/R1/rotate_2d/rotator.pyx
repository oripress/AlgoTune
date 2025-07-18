import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos, floor

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rotate_image(double[:, :] image, double angle_deg, double[:, :] output):
    cdef int n = image.shape[0]
    cdef double center = (n - 1) / 2.0
    cdef double angle_rad = angle_deg * 3.141592653589793 / 180.0
    cdef double cos_a = cos(angle_rad)
    cdef double sin_a = sin(angle_rad)
    
    cdef int i, j
    cdef double x, y, x_orig, y_orig
    cdef int x0, y0, x1, y1
    cdef double dx, dy
    
    for i in range(n):
        for j in range(n):
            # Transform coordinates to center-based
            x = j - center
            y = i - center
            
            # Apply inverse rotation
            x_orig = x * cos_a + y * sin_a + center
            y_orig = -x * sin_a + y * cos_a + center
            
            # Skip if outside image
            if x_orig < 0 or x_orig >= n or y_orig < 0 or y_orig >= n:
                output[i, j] = 0.0
                continue
                
            # Bilinear interpolation
            x0 = <int>floor(x_orig)
            y0 = <int>floor(y_orig)
            x1 = x0 + 1
            y1 = y0 + 1
            
            if x1 >= n or y1 >= n:
                # Use nearest neighbor for edges
                output[i, j] = image[<int>(y_orig+0.5), <int>(x_orig+0.5)]
                continue
                
            dx = x_orig - x0
            dy = y_orig - y0
            
            output[i, j] = (image[y0, x0] * (1-dx) * (1-dy) +
                            image[y0, x1] * dx * (1-dy) +
                            image[y1, x0] * (1-dx) * dy +
                            image[y1, x1] * dx * dy)