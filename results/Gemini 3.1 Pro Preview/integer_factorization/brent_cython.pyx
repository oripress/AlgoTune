# cython: language_level=3
# Final compilation trigger
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cdef extern from *:
    """
    typedef unsigned __int128 uint128_t;
    """
    ctypedef unsigned long long uint128_t

cdef inline unsigned long long gcd(unsigned long long a, unsigned long long b) nogil:
    cdef unsigned long long temp
    while b != 0:
        temp = b
        b = a % b
        a = temp
    return a

cdef inline unsigned long long mod_diff(unsigned long long a, unsigned long long b, unsigned long long n) nogil:
    return a - b if a >= b else a + n - b

cdef inline unsigned long long montgomery_multiply(unsigned long long a, unsigned long long b, unsigned long long n, unsigned long long n_prime) nogil:
    cdef uint128_t T = <uint128_t>a * b
    cdef unsigned long long T_lo = <unsigned long long>T
    cdef unsigned long long T_hi = <unsigned long long>(T >> 64)
    
    cdef unsigned long long m = T_lo * n_prime
    cdef uint128_t mn = <uint128_t>m * n
    cdef unsigned long long mn_hi = <unsigned long long>(mn >> 64)
    
    cdef unsigned long long carry = 1 if T_lo != 0 else 0
    cdef uint128_t t = <uint128_t>T_hi + mn_hi + carry
    
    if t >= n:
        t -= n
    return <unsigned long long>t

cpdef unsigned long long brent(unsigned long long n):
    if n % 2 == 0: return 2
    
    cdef unsigned long long n_prime = n
    n_prime = n_prime * (2 - n * n_prime)
    n_prime = n_prime * (2 - n * n_prime)
    n_prime = n_prime * (2 - n * n_prime)
    n_prime = n_prime * (2 - n * n_prime)
    n_prime = n_prime * (2 - n * n_prime)
    n_prime = -n_prime

    cdef unsigned long long R = (<uint128_t>1 << 64) % n
    cdef unsigned long long R2 = (<uint128_t>R * R) % n
    
    cdef unsigned long long y = montgomery_multiply(2, R2, n, n_prime)
    cdef unsigned long long c = montgomery_multiply(1, R2, n, n_prime)
    cdef unsigned long long m = 1024
    cdef unsigned long long g = 1
    cdef unsigned long long r = 1
    cdef unsigned long long q = R
    cdef unsigned long long x = y
    cdef unsigned long long ys = y
    cdef unsigned long long k = 0
    cdef unsigned long long i = 0
    cdef unsigned long long limit = 0
    
    while g == 1:
        x = y
        for i in range(r):
            y = montgomery_multiply(y, y, n, n_prime)
            y = y + c
            if y >= n: y -= n
        k = 0
        while k < r and g == 1:
            ys = y
            limit = m if m < r - k else r - k
            for i in range(limit):
                y = montgomery_multiply(y, y, n, n_prime)
                y = y + c
                if y >= n: y -= n
                q = montgomery_multiply(q, mod_diff(x, y, n), n, n_prime)
            g = gcd(montgomery_multiply(q, 1, n, n_prime), n)
            k += m
        r *= 2
    if g == n:
        while True:
            ys = montgomery_multiply(ys, ys, n, n_prime)
            ys = ys + c
            if ys >= n: ys -= n
            g = gcd(montgomery_multiply(mod_diff(x, ys, n), 1, n, n_prime), n)
            if g > 1:
                break
    return g