import numpy as np
import cmath

class Solver:
    def solve(self, problem, **kwargs):
        """Highly optimized polynomial root finder using Durand-Kerner method."""
        # Trim leading zeros
        i = 0
        n = len(problem)
        while i < n and problem[i] == 0.0:
            i += 1
        if i == n:
            return []
        p = problem[i:]
        degree = len(p) - 1
        
        # Handle trivial cases
        if degree == 0:
            return []
        if degree == 1:
            return [complex(-p[1] / p[0])]
        if degree == 2:
            a, b, c = p
            disc = b**2 - 4*a*c
            if disc >= 0:
                root1 = (-b + np.sqrt(disc)) / (2*a)
                root2 = (-b - np.sqrt(disc)) / (2*a)
                roots = [root1, root2]
            else:
                real_part = -b/(2*a)
                imag_part = np.sqrt(-disc)/(2*a)
                roots = [complex(real_part, imag_part), 
                         complex(real_part, -imag_part)]
            # Sort roots
            return sorted(roots, key=lambda z: (z.real, z.imag), reverse=True)
        
        # Durand-Kerner method for degree >= 3
        p = [complex(c) for c in p]  # Convert to complex
        n = degree
        
        # Initial guesses: roots of unity scaled by magnitude
        max_coeff = max(abs(c) for c in p[1:])
        r = 2 * max_coeff / abs(p[0])  # Radius estimate
        roots = [r * cmath.exp(2j * cmath.pi * k / n) for k in range(n)]
        
        # Iterate until convergence
        max_iter = 100
        tol = 1e-10
        for _ in range(max_iter):
            new_roots = []
            max_diff = 0.0
            for i in range(n):
                denom = 1.0
                term = p[0] * roots[i] + p[1]  # Start with linear terms
                for j in range(n):
                    if i != j:
                        denom *= (roots[i] - roots[j])
                # Evaluate polynomial using Horner's method
                value = p[0]
                for k in range(1, len(p)):
                    value = value * roots[i] + p[k]
                # Update root
                new_root = roots[i] - value / denom
                new_roots.append(new_root)
                max_diff = max(max_diff, abs(new_root - roots[i]))
            roots = new_roots
            if max_diff < tol:
                break
        
        # Sort roots
        sorted_roots = sorted(roots, key=lambda z: (z.real, z.imag), reverse=True)
        return sorted_roots