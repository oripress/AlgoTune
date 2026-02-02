import math
from sympy.ntheory import factorint
from sympy.ntheory.residue_ntheory import discrete_log as sympy_discrete_log

class Solver:
    def solve(self, problem, **kwargs):
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        
        # Trivial cases
        if h == 1:
            return {"x": 0}
        if h == g:
            return {"x": 1}
        if h == g * g % p:
            return {"x": 2}
        
        try:
            x = self._solve_pohlig_hellman(p, g, h)
            # Verify solution
            if pow(g, x, p) == h:
                return {"x": x}
        except:
            pass
        
        # Fall back to sympy
        return {"x": sympy_discrete_log(p, h, g)}
    
    def _solve_pohlig_hellman(self, p, g, h):
        n = p - 1  # order of the group
        factors = factorint(n)
        
        results = []
        moduli = []
        
        for prime, exp in factors.items():
            pe = prime ** exp
            gi = pow(g, n // pe, p)
            hi = pow(h, n // pe, p)
            xi = self._solve_prime_power(gi, hi, p, prime, exp)
            results.append(xi)
            moduli.append(pe)
        
        return self._crt(results, moduli)
    
    def _bsgs(self, g, h, p, n):
        """Baby-step Giant-step algorithm"""
        m = math.isqrt(n) + 1
        
        # Baby step: build table of g^j mod p
        table = {}
        val = 1
        for j in range(m):
            if val == h:
                return j
            table[val] = j
            val = (val * g) % p
        
        # Giant step: g^(-m) mod p
        factor = pow(g, n - m, p)  # g^(-m) = g^(n-m) since g^n = 1
        gamma = h
        for i in range(m):
            if gamma in table:
                return i * m + table[gamma]
            gamma = (gamma * factor) % p
        
        return None  # No solution found
    
    def _solve_prime_power(self, g, h, p, q, e):
        """Solve g^x = h in a group of order q^e using Pohlig-Hellman for prime powers"""
        x = 0
        n_order = q ** e
        gamma = pow(g, q ** (e - 1), p)  # has order q
        
        for k in range(e):
            # Compute h_k = (h * g^(-x))^(q^{e-1-k})
            if x > 0:
                g_inv_x = pow(g, n_order - (x % n_order), p)
                hk = pow((h * g_inv_x) % p, q ** (e - 1 - k), p)
            else:
                hk = pow(h, q ** (e - 1 - k), p)
            
            # Solve gamma^dk = hk
            if q <= 1000:
                dk = self._brute_force(gamma, hk, p, q)
            else:
                dk = self._bsgs(gamma, hk, p, q)
            
            if dk is None:
                raise ValueError("No solution found in subproblem")
            
            x += dk * (q ** k)
        
        return x
    
    def _brute_force(self, g, h, p, n):
        """Brute force for small n"""
        val = 1
        for j in range(n):
            if val == h:
                return j
            val = (val * g) % p
        return None
    
    def _crt(self, remainders, moduli):
        """Chinese Remainder Theorem"""
        M = 1
        for m in moduli:
            M *= m
        
        result = 0
        for ri, mi in zip(remainders, moduli):
            Mi = M // mi
            inv = pow(Mi, -1, mi)
            result += ri * Mi * inv
        
        return result % M