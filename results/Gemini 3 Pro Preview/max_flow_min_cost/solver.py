import solver_cython

class Solver:
    def solve(self, problem, **kwargs):
        # Convert list of lists to numpy arrays if they aren't already, 
        # though the cython code iterates over them assuming they are indexable.
        # The cython function expects the problem dict.
        return solver_cython.solve_cython(problem)