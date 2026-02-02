import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        coefficients = problem
        computed_roots = np.roots(coefficients)
        sorted_roots = sorted(computed_roots, key=lambda z: (z.real, z.imag), reverse=True)
        # Return as list of complex numbers
        return [complex(r) for r in sorted_roots]