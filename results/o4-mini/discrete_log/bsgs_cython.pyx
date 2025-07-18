# cython: language_level=3
cimport libc.math
from libcpp.unordered_map cimport unordered_map
cimport cython

ctypedef unsigned long long ull

@cython.boundscheck(False)
@cython.wraparound(False)
cdef ull mod_pow(ull base, ull exp, ull mod):
    cdef ull result = 1
    cdef ull e = exp
    base %= mod
    while e > 0:
        if e & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        e >>= 1
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ull c_bsgs(ull g, ull h, ull p):
    cdef ull n = p - 1
    cdef double s = libc.math.sqrt(n)
    cdef ull m = <ull>s + 1
    cdef unordered_map[ull, ull] baby = unordered_map[ull, ull]()
    baby.reserve(m)
    cdef ull aj = 1
    cdef ull j
    for j in range(m):
        if baby.find(aj) == baby.end():
            baby[aj] = j
        aj = (aj * g) % p
    cdef ull inv_g = mod_pow(g, p - 2, p)
    cdef ull factor = mod_pow(inv_g, m, p)
    cdef ull gamma = h
    cdef ull i
    for i in range(m):
        it = baby.find(gamma)
        if it != baby.end():
            return i * m + it.second
        gamma = (gamma * factor) % p
    return <ull>(-1)