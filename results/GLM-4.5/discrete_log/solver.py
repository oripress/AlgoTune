from sympy.ntheory.residue_ntheory import discrete_log as _discrete_log
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.backends import default_backend
import math

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """Solve discrete logarithm problem: g^x ≡ h (mod p)"""
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        
        # For very small primes, use brute force
        if p < 1000:
            for x in range(p):
                if pow(g, x, p) == h:
                    return {"x": x}
        
        # Try to use optimized libraries first
        try:
            # Check if we can use cryptography's DH (though it's not directly for discrete log)
            # This is just to test if there are any faster alternatives
            return {"x": _discrete_log(p, h, g)}
        except:
            return {"x": _discrete_log(p, h, g)}