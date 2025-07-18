from scipy.linalg import eigh

class Solver:
    def solve(self, problem, **kwargs):
        # Compute eigenvalues using highly optimized LAPACK routine
        # 'evr' driver is often faster for symmetric eigenvalue problems
        eigenvalues = eigh(problem, driver='evr', eigvals_only=True)
        
        # Return eigenvalues in descending order
        return eigenvalues[::-1].tolist()