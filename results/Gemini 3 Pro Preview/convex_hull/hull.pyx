# distutils: language = c++
import numpy as np
cimport numpy as np
cimport cython
from libcpp.algorithm cimport sort
from libcpp cimport bool
from libc.stdlib cimport malloc, free

cdef struct Point:
    double x
    double y
    int index

cdef bool compare_points(const Point& a, const Point& b) noexcept:
    if a.x < b.x:
        return True
    elif a.x > b.x:
        return False
    else:
        return a.y < b.y

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_hull(double[:, :] points):
    cdef int n = points.shape[0]
    if n == 0:
        return [], []
        
    # Allocate memory for sorted_points, lower, and upper in one go
    cdef Point* memory_block = <Point*>malloc(3 * n * sizeof(Point))
    
    if memory_block == NULL:
        raise MemoryError()
        
    cdef Point* sorted_points = memory_block
    cdef Point* lower = memory_block + n
    cdef Point* upper = memory_block + 2 * n
    
    cdef int i
    cdef int k = 0
    cdef int t = 0
    cdef Point p, p1, p2
    cdef double val
    cdef int limit_lower, limit_upper, total_len, idx, j
    cdef np.ndarray[np.int64_t, ndim=1] vertices_arr
    cdef np.ndarray[np.float64_t, ndim=2] points_arr
    
    try:
        for i in range(n):
            sorted_points[i].x = points[i, 0]
            sorted_points[i].y = points[i, 1]
            sorted_points[i].index = i
            
        sort(sorted_points, sorted_points + n, compare_points)
        
        # Lower hull
        for i in range(n):
            p = sorted_points[i]
            while k >= 2:
                p1 = lower[k-1]
                p2 = lower[k-2]
                val = (p1.x - p2.x) * (p.y - p2.y) - (p1.y - p2.y) * (p.x - p2.x)
                if val <= 0:
                    k -= 1
                else:
                    break
            lower[k] = p
            k += 1
            
        # Upper hull
        for i in range(n - 1, -1, -1):
            p = sorted_points[i]
            while t >= 2:
                p1 = upper[t-1]
                p2 = upper[t-2]
                val = (p1.x - p2.x) * (p.y - p2.y) - (p1.y - p2.y) * (p.x - p2.x)
                if val <= 0:
                    t -= 1
                else:
                    break
            upper[t] = p
            t += 1
            
        limit_lower = k - 1
        if limit_lower < 0: limit_lower = 0
        
        limit_upper = t - 1
        if limit_upper < 0: limit_upper = 0
        
        total_len = limit_lower + limit_upper
        
        vertices_arr = np.empty(total_len, dtype=np.int64)
        points_arr = np.empty((total_len, 2), dtype=np.float64)
        
        idx = 0
        
        for j in range(limit_lower):
            vertices_arr[idx] = lower[j].index
            points_arr[idx, 0] = lower[j].x
            points_arr[idx, 1] = lower[j].y
            idx += 1
            
        for j in range(limit_upper):
            vertices_arr[idx] = upper[j].index
            points_arr[idx, 0] = upper[j].x
            points_arr[idx, 1] = upper[j].y
            idx += 1
            
        return vertices_arr.tolist(), points_arr.tolist()
        
    finally:
        free(memory_block)