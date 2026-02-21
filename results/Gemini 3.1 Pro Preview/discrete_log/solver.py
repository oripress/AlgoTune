import sympy
import math
from sympy.ntheory.residue_ntheory import discrete_log

def bsgs(g, h, p, q):
    m = math.isqrt(q) + 1
    table = {}
    e = 1
    for i in range(m):
        table[e] = i
        e = (e * g) % p
        
    factor = pow(g, -m, p)
    e = h
    for j in range(m):
        if e in table:
            return j * m + table[e]
        e = (e * factor) % p
    return -1

def crt(remainders, moduli):
    total = 0
    prod = 1
    for m in moduli:
        prod *= m
    for r, m in zip(remainders, moduli):
        p_i = prod // m
        total += r * p_i * pow(p_i, -1, m)
    return total % prod

class Solver:
    def solve(self, problem: dict[str, int], **kwargs) -> dict[str, int]:
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        
        # Try Pohlig-Hellman assuming g is a primitive root
        try:
            factors = sympy.factorint(p - 1)
            
            remainders = []
            moduli = []
            
            for q, e in factors.items():
                qe = q**e
                moduli.append(qe)
                
                x_q = 0
                current_h = h
                
                g_q = pow(g, (p - 1) // q, p)
                
                for k in range(e):
                    h_k = pow(current_h, (p - 1) // (q**(k + 1)), p)
                    z_k = bsgs(g_q, h_k, p, q)
                    if z_k == -1:
                        raise ValueError("No solution found in subgroup")
                    x_q += z_k * (q**k)
                    
                    inv_g = pow(g, -(z_k * (q**k)), p)
                    current_h = (current_h * inv_g) % p
                    
                remainders.append(x_q)
                
            x = crt(remainders, moduli)
            if pow(g, x, p) == h:
                return {"x": x}
        except Exception:
            pass
            
        return {"x": discrete_log(p, h, g)}