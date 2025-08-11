import math
from typing import Any
import numba as nb
from numba import types
from numba.typed import Dict
import numpy as np
from sympy.ntheory.residue_ntheory import discrete_log

@nb.jit(nopython=True, cache=True)
def modpow(base, exp, mod):
    """Fast modular exponentiation."""
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

@nb.jit(nopython=True, cache=True)
def baby_giant_step_numba(p, g, h):
    """
    JIT-compiled baby-step giant-step algorithm for discrete logarithm.
    Find x such that g^x ≡ h (mod p)
    """
    # Handle trivial cases
    if h == 1:
        return 0
    if h == g:
        return 1
    
    # Baby-step giant-step algorithm
    m = int(np.ceil(np.sqrt(p - 1)))
    
    # Baby steps: compute g^j mod p for j = 0, 1, ..., m-1
    # Use typed dictionary for O(1) lookups
    baby_steps = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )
    
    current = 1
    for j in range(m):
        if current == h:
            return j
        baby_steps[current] = j
        current = (current * g) % p
    
    # Giant steps: compute h * (g^{-m})^i mod p for i = 0, 1, ..., m-1
    # First compute g^{-m} mod p using Fermat's little theorem
    g_inv_m = modpow(g, p - 1 - m, p)
    
    gamma = h
    for i in range(m):
        if gamma in baby_steps:
            x = (i * m + baby_steps[gamma]) % (p - 1)
            # Verify the result
            if modpow(g, x, p) == h:
                return x
        gamma = (gamma * g_inv_m) % p
    
    # If not found, return -1 to indicate failure
    return -1

class Solver:
    def __init__(self):
        """Precompute any necessary data structures."""
        # Trigger JIT compilation with a dummy call
        _ = baby_giant_step_numba(23, 5, 8)
    
    def solve(self, problem: dict[str, int], **kwargs) -> dict[str, int]:
        """
        Solve the discrete logarithm problem using hybrid approach.
        """
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        
        # Use numba implementation for smaller primes
        if p < 10**9:
            x = baby_giant_step_numba(p, g, h)
            if x >= 0:
                return {"x": int(x)}
        
        # Fall back to sympy for larger primes or if numba fails
        return {"x": discrete_log(p, h, g)}