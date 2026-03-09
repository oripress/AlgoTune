# cython: language_level=3
import math

cpdef object factor_semiprime(object n):
    cdef object gcd = math.gcd
    cdef object y, c, x, ys, q, diff, g, d
    cdef int attempt
    cdef object seed

    if n % 2 == 0:
        return 2, n // 2
    if n % 3 == 0:
        return 3, n // 3
    if n % 5 == 0:
        return 5, n // 5

    seed = (n & 127) + 1
    for attempt in range(1, 10):
        y = attempt + 1
        c = seed + (attempt << 1) - 1
        if c >= n:
            c %= n
        if c == 0:
            c = 1

        g = 1
        r = 1
        m = 128

        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n

            q = 1
            k = 0
            while k < r and g == 1:
                ys = y
                limit = m if m < (r - k) else (r - k)
                for _ in range(limit):
                    y = (y * y + c) % n
                    diff = x - y
                    if diff < 0:
                        diff = -diff
                    q = (q * diff) % n
                g = gcd(q, n)
                k += limit
            r <<= 1

        if g == n:
            while True:
                ys = (ys * ys + c) % n
                diff = x - ys
                if diff < 0:
                    diff = -diff
                g = gcd(diff, n)
                if g > 1:
                    break

        if 1 < g < n:
            d = g
            break
    else:
        d = n
        for c in (1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43):
            x = 2
            y = 2
            g = 1
            while g == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                diff = x - y
                if diff < 0:
                    diff = -diff
                g = gcd(diff, n)
            if g != n:
                d = g
                break

    if d == n:
        return 1, n
    if d > n // d:
        return n // d, d
    return d, n // d