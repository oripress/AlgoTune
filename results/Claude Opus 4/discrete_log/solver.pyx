# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from sympy.ntheory.residue_ntheory import discrete_log
cimport cython

cdef class Solver:
    def solve(self, dict problem):
        cdef long p = problem["p"]
        cdef long g = problem["g"]
        cdef long h = problem["h"]
        cdef long x
        
        # For very small p, brute force with optimized modular arithmetic
        if p <= 100:
            x = 1
            for i in range(p):
                if x == h:
                    return {"x": i}
                x = (x * g) % p
        
        # For larger p, use sympy
        return {"x": discrete_log(p, h, g)}