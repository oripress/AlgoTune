from scipy.linalg import lu

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the LU factorization A = P L U using SciPy LU routine.
        Uses overwrite_a and disables input checking for speed.
        """
        # Compute LU factorization with partial pivoting, no input checks
        P, L, U = lu(problem["matrix"], overwrite_a=True, check_finite=False)
        # Return numpy arrays directly to avoid list conversion overhead
        return {"LU": {"P": P, "L": L, "U": U}}