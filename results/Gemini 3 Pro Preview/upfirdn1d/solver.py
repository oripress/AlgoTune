try:
    from solver_cython import solve_cython
except ImportError:
    solve_cython = None

import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: list) -> list:
        if solve_cython is not None:
            return solve_cython(problem)
        
        # Fallback to reference if compilation failed
        results = []
        for h, x, up, down in problem:
            res = signal.upfirdn(h, x, up=up, down=down)
            results.append(res)
        return results