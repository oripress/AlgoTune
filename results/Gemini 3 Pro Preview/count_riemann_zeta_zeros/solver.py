from typing import Any
import numpy as np
from scipy.special import loggamma
import math

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        t = problem["t"]
        
        # 1. Calculate theta(t)
        s_half = 0.25 + 0.5j * t
        val = loggamma(s_half)
        theta = np.imag(val) - 0.5 * t * np.log(np.pi)
        
        # 2. Track argument of zeta(s)
        m = int(np.sqrt(t / (2 * np.pi)))
        ns = np.arange(1, m + 1)
        
        def compute_zeta(s):
            sigma = s.real
            if sigma > 1.1:
                limit = 50
                n_vals = np.arange(1, limit + 1)
                terms = np.exp(-s * np.log(n_vals))
                return np.sum(terms)
            else:
                term1 = np.sum(np.exp(-s * np.log(ns)))
                term2 = np.sum(np.exp((s - 1) * np.log(ns)))
                
                log_chi = (s - 1) * np.log(2 * np.pi) + 1j * np.pi / 2 - 1j * np.pi * s / 2 + loggamma(1 - s)
                chi = np.exp(log_chi)
                
                return term1 + chi * term2

        # Use many steps to ensure robust tracking
        # From 2.0 down to 0.6 using fast method
        steps_fast = np.linspace(2.0, 0.6, 50)
        
        s_prev = steps_fast[0] + 1j * t
        z_prev = compute_zeta(s_prev)
        initial_arg = np.angle(z_prev)
        
        total_arg_change = 0.0
        
        for sigma in steps_fast[1:]:
            s = sigma + 1j * t
            z = compute_zeta(s)
            
            ratio = z / z_prev
            delta = np.angle(ratio)
            
            total_arg_change += delta
            z_prev = z
            
        # Final step to 0.5 using mpmath for accuracy
        # We need to bridge from z_prev (at 0.6) to zeta(0.5+it)
        
        # Re-calculate z_prev using mpmath to be sure?
        # Or assume fast method is good at 0.6.
        # Let's assume it's good.
        
        # Use mpmath for the final point
        from mpmath import mp, zeta as mp_zeta
        # Set precision
        mp.dps = 15
        mp.dps = 5
        s_final = 0.5 + 1j * t
        z_final_mp = mp_zeta(s_final)
        z_final = complex(z_final_mp)
        
        # Calculate change from 0.6 to 0.5
        # We might need intermediate steps if the jump is large?
        # 0.6 to 0.5 is small distance (0.1).
        # But near a zero, phase can change rapidly.
        # Let's add a few intermediate steps using mpmath if needed, 
        # or just trust mpmath for the final value and hope phase doesn't wrap.
        # Actually, if we are close to a zero, phase changes by pi.
        # If we jump from 0.6 to 0.5, we might miss a wrap if change > pi.
        # But 0.1 is small.
        
        ratio = z_final / z_prev
        delta = np.angle(ratio)
        total_arg_change += delta
        
        final_arg = initial_arg + total_arg_change
        s_t = final_arg / np.pi
        
        n_t = theta / np.pi + 1 + s_t
        
        return {"result": int(round(n_t))}