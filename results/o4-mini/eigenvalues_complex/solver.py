from numpy.linalg import eigvals

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute eigenvalues of a real square matrix and return them sorted
        in descending order by real part then imaginary part.
        """
        # Directly compute eigenvalues (no eigenvectors)
        ev = eigvals(problem)
        # Sort ascending by real then imaginary part (NumPy default for complex)
        ev.sort()
        # Reverse for descending
        return ev[::-1].tolist()