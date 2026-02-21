# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport floor, fabs
import scipy.stats as stats

cdef double compute_prob_outside_square(int n, int h):
    if h == 0:
        return 1.0
    cdef double P = 0.0
    cdef int k = n // h
    cdef double p1
    cdef int j
    while k >= 0:
        p1 = 1.0
        for j in range(h):
            p1 = (n - k * h - j) * p1 / (n + k * h + j + 1)
        P = p1 * (1.0 - P)
        k -= 1
    cdef double prob = 2 * P
    if prob > 1.0:
        return 1.0
    elif prob < 0.0:
        return 0.0
    return prob

cdef int compute_ks_stat(double[:] data1, double[:] data2, int n):
    cdef int i = 0
    cdef int j = 0
    cdef int max_d = 0
    cdef int d
    cdef double val
    
    while i < n and j < n:
        if data1[i] < data2[j]:
            val = data1[i]
            while i < n and data1[i] == val:
                i += 1
        elif data1[i] > data2[j]:
            val = data2[j]
            while j < n and data2[j] == val:
                j += 1
        else:
            val = data1[i]
            while i < n and data1[i] == val:
                i += 1
            while j < n and data2[j] == val:
                j += 1
        
        d = i - j
        if d < 0:
            d = -d
        if d > max_d:
            max_d = d
            
    if i < n:
        d = n - i
        if d > max_d:
            max_d = d
    elif j < n:
        d = n - j
        if d > max_d:
            max_d = d
            
    return max_d

def solve_cython(sample1, sample2):
    cdef int n = len(sample1)
    cdef np.ndarray[np.float64_t, ndim=1] arr1 = np.sort(np.asarray(sample1, dtype=np.float64))
def solve_cython(sample1, sample2):
    cdef int n = len(sample1)
    sample1.sort()
    sample2.sort()
    cdef np.ndarray[np.float64_t, ndim=1] arr1 = np.asarray(sample1, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] arr2 = np.asarray(sample2, dtype=np.float64)
    
    cdef int h = compute_ks_stat(arr1, arr2, n)
    cdef double d = <double>h / n
    cdef double prob
    
    if n <= 10000:
        prob = compute_prob_outside_square(n, h)
    else:
        prob = stats.distributions.kstwo.sf(d, n / 2.0)
        if prob > 1.0:
            prob = 1.0
        elif prob < 0.0:
            prob = 0.0
            
    return {"statistic": d, "pvalue": prob}