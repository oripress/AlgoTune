import math
import random

# Miller-Rabin primality test
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    # small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for sp in small_primes:
        if n % sp == 0:
            return n == sp
    d = n - 1
    s = 0
    while d & 1 == 0:
        d >>= 1
        s += 1
    # bases for deterministic test up to 2^64
    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

# Pollard's Rho factorization
def pollards_rho(n: int) -> int:
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        c = random.randrange(1, n - 1)
        x = random.randrange(2, n - 1)
        y = x
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
            if d == n:
                break
        if d > 1 and d < n:
            return d

# Factorization using Pollard's Rho
def factorint(n: int, res=None) -> dict:
    if res is None:
        res = {}
    if n == 1:
        return res
    if is_prime(n):
        res[n] = res.get(n, 0) + 1
    else:
        d = pollards_rho(n)
        factorint(d, res)
        factorint(n // d, res)
    return res

# Baby-step Giant-step algorithm for discrete logarithm
def bsgs(base: int, target: int, p: int, order: int) -> int:
    m = int(math.isqrt(order)) + 1
    table = {}
    baby = 1
    for j in range(m):
        if baby not in table:
            table[baby] = j
        baby = baby * base % p
    # base^{-m} mod p via Fermat
    factor = pow(base, p - 1 - m, p)
    gamma = target
    for i in range(m):
        if gamma in table:
            return i * m + table[gamma]
        gamma = gamma * factor % p
    return None

# Discrete log modulo prime power q^e
def dlp_prime_power(g: int, h: int, p: int, q: int, e: int) -> int:
    # Pohlig-Hellman lifting for prime power q^e
    N = p - 1
    # base for order q
    g1 = pow(g, N // q, p)
    # initial discrete log mod q
    h1 = pow(h, N // q, p)
    x = bsgs(g1, h1, p, q)
    if x is None:
        return None
    result = x
    # lift to higher powers
    for k in range(1, e):
        exp = N // (q ** (k + 1))
        # compute numerator h * g^{-result} mod p
        numerator = h * pow(g, -result, p) % p
        h_sub = pow(numerator, exp, p)
        d = bsgs(g1, h_sub, p, q)
        if d is None:
            return None
        result += d * (q ** k)
    return result

# Chinese Remainder Theorem
def crt(congruences: list) -> int:
    x = 0
    M = 1
    for ai, ni in congruences:
        ai_mod = ai % ni
        inv = pow(M, -1, ni)
        k = (ai_mod - x) * inv % ni
        x += M * k
        M *= ni
    return x % M

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        if h == 1:
            return {"x": 0}
        N = p - 1
        # Factor group order
        facs = factorint(N)
        congruences = []
        for q, e in facs.items():
            m = q ** e
            xi = dlp_prime_power(g, h, p, q, e)
            if xi is None:
                # fallback to general BSGS
                x0 = bsgs(g, h, p, N)
                return {"x": x0}
            congruences.append((xi, m))
        # Combine via CRT
        x = crt(congruences)
        return {"x": x}