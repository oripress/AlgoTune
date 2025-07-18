import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def kmeans(np.ndarray[np.float64_t, ndim=2] X, int k, int max_iter=300, double tol=1e-4):
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] centroids = X[np.random.choice(n, k, replace=False), :].copy()
    cdef np.ndarray[np.int_t, ndim=1] labels = np.zeros(n, dtype=np.int_)
    cdef np.ndarray[np.float64_t, ndim=1] distances = np.zeros(n)
    cdef double diff, dist, min_dist
    cdef int i, j, idx, iter_count, min_idx
    cdef np.ndarray[np.float64_t, ndim=2] new_centroids
    cdef np.ndarray[np.int_t, ndim=1] counts
    
    for iter_count in range(max_iter):
        # Assign labels
        for i in range(n):
            min_dist = 1e20
            min_idx = -1
            for j in range(k):
                dist = 0.0
                for idx in range(d):
                    diff = X[i, idx] - centroids[j, idx]
                    dist += diff * diff
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
            labels[i] = min_idx
            distances[i] = min_dist
        
        # Update centroids
        new_centroids = np.zeros((k, d))
        counts = np.zeros(k, dtype=np.int_)
        for i in range(n):
            for idx in range(d):
                new_centroids[labels[i], idx] += X[i, idx]
            counts[labels[i]] += 1
        
        # Check for convergence
        diff = 0.0
        for j in range(k):
            if counts[j] > 0:
                for idx in range(d):
                    new_centroids[j, idx] /= counts[j]
                    diff += (new_centroids[j, idx] - centroids[j, idx]) ** 2
            else:
                # Reinitialize empty centroid
                new_centroids[j] = X[np.random.randint(n)]
        
        if sqrt(diff) < tol:
            break
        
        centroids = new_centroids
    
    return labels