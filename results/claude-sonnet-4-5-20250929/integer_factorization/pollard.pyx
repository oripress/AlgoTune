# cython: language_level=3
import math

def pollard_rho_cython(n):
    """Optimized Pollard's rho in Cython"""
    cdef long long c, x, y, d, product, count, batch_size
    
    for c in [1, 2, 3]:
        x = y = 2
        d = 1
        product = 1
        count = 0
        batch_size = 2000
        
        while d == 1 and count < 1000000:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            
            product = (product * abs(x - y)) % n
            count += 1
            
            if count % batch_size == 0:
                d = math.gcd(product, n)
                if d != 1:
                    break
                product = 1
        
        if d == 1 and count % batch_size != 0:
            d = math.gcd(product, n)
        
        if 1 < d < n:
            return int(d)
    
    return None