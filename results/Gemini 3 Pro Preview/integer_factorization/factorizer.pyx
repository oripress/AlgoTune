# cython: language_level=3, boundscheck=False, wraparound=False
from math import gcd

cdef extern from *:
    """
    static unsigned long long mul_mod(unsigned long long a, unsigned long long b, unsigned long long m) {
        unsigned __int128 res = (unsigned __int128)a * b;
        return (unsigned long long)(res % m);
    }
    """
    unsigned long long mul_mod(unsigned long long a, unsigned long long b, unsigned long long m)

cdef unsigned long long gcd_c(unsigned long long a, unsigned long long b):
    cdef unsigned long long t
    while b:
        t = b
        b = a % b
        a = t
    return a

cdef unsigned long long pollard_rho_brent_c(unsigned long long n):
    if n % 2 == 0:
        return 2
    
    cdef unsigned long long y = 2
    cdef unsigned long long c = 1
    cdef unsigned long long m = 128
    cdef unsigned long long g = 1
    cdef unsigned long long r = 1
    cdef unsigned long long q = 1
    cdef unsigned long long x = 2
    cdef unsigned long long ys = 2
    
    cdef unsigned long long i, k, limit
    cdef unsigned long long diff
    
    while g == 1:
        x = y
        for i in range(r):
            y = mul_mod(y, y, n)
            y = (y + c) % n
        
        k = 0
        while k < r and g == 1:
            ys = y
            limit = m
            if r - k < m:
                limit = r - k
            
            for i in range(limit):
                y = mul_mod(y, y, n)
                y = (y + c) % n
                
                if x > y:
                    diff = x - y
                else:
                    diff = y - x
                
                q = mul_mod(q, diff, n)
                
            g = gcd_c(q, n)
            k += m
        r *= 2
        
    if g == n:
        while True:
            y = mul_mod(ys, ys, n)
            ys = (y + c) % n
            
            if x > ys:
                diff = x - ys
            else:
                diff = ys - x
                
            g = gcd_c(diff, n)
            if g > 1:
                break
    
    return g

def pollard_rho_brent_obj(n):
    if n % 2 == 0:
        return 2
    
    cdef object y = 2
    cdef object c = 1
    cdef long m = 128
    cdef object g = 1
    cdef long r = 1
    cdef object q = 1
    cdef object x = 2
    cdef object ys = 2
    
    cdef long i
    cdef long k
    cdef long limit
    
    while g == 1:
        x = y
        for i in range(r):
            y = (y*y + c) % n
        
        k = 0
        while k < r and g == 1:
            ys = y
            limit = m
            if r - k < m:
                limit = r - k
            
            for i in range(limit):
                y = (y*y + c) % n
                q = (q * abs(x - y)) % n
            g = gcd(q, n)
            k += m
        r *= 2
        
    if g == n:
        while True:
            ys = (ys*ys + c) % n
            g = gcd(abs(x - ys), n)
            if g > 1:
                break
    
    return g

    return g

cdef unsigned long long small_primes[15]
small_primes[:] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def pollard_rho_brent(n):
    # Trial division for small primes
    cdef int i
    cdef unsigned long long p
    
    # Check small primes first
    for i in range(15):
        p = small_primes[i]
        if n % p == 0:
            return p
            
    # Check if n fits in 64-bit unsigned integer
    # Max value is 2^64 - 1 = 18446744073709551615
    if n < 18446744073709551615:
        return pollard_rho_brent_c(n)
    else:
        return pollard_rho_brent_obj(n)