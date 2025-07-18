import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """Optimized linear system solver using NumPy with explicit dtype and overwrite options."""
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        x = np.linalg.solve(A, b)
        return x.tolist()