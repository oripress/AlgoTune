import numpy as np
from numpy.typing import NDArray
from typing import Any
try:
    from solver_cython import solve_cython
except ImportError:
    solve_cython = None

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> Any:
        if solve_cython is not None:
            # Ensure the array is C-contiguous and float64
            if not problem.flags['C_CONTIGUOUS']:
                problem = np.ascontiguousarray(problem)
            if problem.dtype != np.float64:
                problem = problem.astype(np.float64)
            return solve_cython(problem)
        
        # Fallback implementation
        n = problem.shape[0]
        
        if n == 1:
            return [complex(problem[0, 0], 0.0)]
            
        if n == 2:
            a = problem[0, 0]
            b = problem[0, 1]
            c = problem[1, 0]
            d = problem[1, 1]
            
            trace = a + d
            det = a*d - b*c
            
            delta = trace*trace - 4*det
            sqrt_delta = (delta + 0j)**0.5
            
            l1 = (trace + sqrt_delta) / 2.0
            l2 = (trace - sqrt_delta) / 2.0
            
            if l1.real > l2.real:
                return [l1, l2]
            elif l2.real > l1.real:
                return [l2, l1]
            else:
                if l1.imag >= l2.imag:
                    return [l1, l2]
                else:
                    return [l2, l1]

        if np.array_equal(problem, problem.T):
             eigenvalues = np.linalg.eigvalsh(problem)
             return [complex(x, 0.0) for x in eigenvalues[::-1]]
        else:
             eigenvalues = np.linalg.eigvals(problem)
             eigenvalues.sort()
             return eigenvalues[::-1].tolist()