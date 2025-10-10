# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.math cimport sqrt, ceil
from libc.stdlib cimport malloc, free

cdef long long mod_pow(long long base, long long exp, long long mod) nogil:
    """Fast modular exponentiation."""
    cdef long long result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

cpdef long long baby_step_giant_step(long long g, long long h, long long p):
    """Optimized baby-step giant-step algorithm."""
    cdef long long m = <long long>ceil(sqrt(<double>p))
    cdef dict table = {}
    cdef long long g_power = 1
    cdef long long j, i, g_m, g_inv_m, gamma
    
    # Baby step
    for j in range(m):
        if g_power == h:
            return j
        table[g_power] = j
        g_power = (g_power * g) % p
    
    # Giant step
    g_m = mod_pow(g, m, p)
    g_inv_m = mod_pow(g_m, p - 2, p)  # Fermat's little theorem
    
    gamma = h
    for i in range(m):
        if gamma in table:
            return i * m + table[gamma]
        gamma = (gamma * g_inv_m) % p
    
    return -1