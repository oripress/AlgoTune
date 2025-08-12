import numpy as np
try:
    from solver_cython import solve_quadratic, solve_linear, sort_roots
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    from scipy.linalg import eigvals
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import dace
    DACE_AVAILABLE = True
except ImportError:
    DACE_AVAILABLE = False

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def sort_roots_numba(roots_real, roots_imag):
        """Sort roots by real part, then imaginary part, descending."""
        n = len(roots_real)
        for i in range(n):
            for j in range(i + 1, n):
                if roots_real[i] < roots_real[j] or (roots_real[i] == roots_real[j] and roots_imag[i] < roots_imag[j]):
                    roots_real[i], roots_real[j] = roots_real[j], roots_real[i]
                    roots_imag[i], roots_imag[j] = roots_imag[j], roots_imag[i]
        return roots_real, roots_imag

def build_companion_matrix_fast(coefficients):
    """Build companion matrix more efficiently."""
    n = len(coefficients) - 1
    companion = np.zeros((n, n), dtype=np.float64)
    # Set the subdiagonal
    for i in range(n-1):
        companion[i+1, i] = 1.0
    # Set the last row
    companion[-1, :] = -np.array(coefficients[:-1]) / coefficients[0]
    return companion

if DACE_AVAILABLE:
    @dace.program
    def build_companion_matrix_dace(coefficients):
        """Build companion matrix using DaCe for optimization."""
        n = len(coefficients) - 1
        companion = np.zeros((n, n), dtype=np.float64)
        # Set the subdiagonal
        for i in range(n-1):
            companion[i+1, i] = 1.0
        # Set the last row
        companion[-1, :] = -np.array(coefficients[:-1]) / coefficients[0]
        return companion

def find_roots_final_optimized(coefficients):
    """Find roots using final optimized methods."""
    n = len(coefficients) - 1
    
    if n <= 2:
        # For very small polynomials, use direct methods
        return np.roots(coefficients).tolist()
    elif n <= 5:
        # For small polynomials, try DaCe optimized companion matrix
        if DACE_AVAILABLE:
            try:
                companion = build_companion_matrix_dace(np.array(coefficients))
                if SCIPY_AVAILABLE:
                    eigenvalues = eigvals(companion)
                else:
                    eigenvalues = np.linalg.eigvals(companion)
                return eigenvalues.tolist()
            except:
                pass
        
        # Fallback to standard companion matrix
        companion = build_companion_matrix_fast(coefficients)
        if SCIPY_AVAILABLE:
            eigenvalues = eigvals(companion)
        else:
            eigenvalues = np.linalg.eigvals(companion)
        return eigenvalues.tolist()
    elif n <= 10:
        # For medium polynomials, use optimized companion matrix
        companion = build_companion_matrix_fast(coefficients)
        if SCIPY_AVAILABLE:
            eigenvalues = eigvals(companion)
        else:
            eigenvalues = np.linalg.eigvals(companion)
        return eigenvalues.tolist()
    else:
        # For larger polynomials, use numpy.roots
        return np.roots(coefficients).tolist()

class Solver:
    def solve(self, problem, **kwargs) -> list[complex]:
        """
        Solve the polynomial problem by finding all roots of the polynomial.

        The polynomial is given as a list of coefficients [a_n, a_{n-1}, ..., a_0],
        representing p(x) = a_n * x^n + a_{n-1} * x^{n-1} + ... + a_0.
        The coefficients are real numbers.
        This method computes all the roots (which may be real or complex) and returns them
        sorted in descending order by their real parts and, if necessary, by their imaginary parts.

        :param problem: A list of polynomial coefficients (real numbers) in descending order.
        :return: A list of roots (real and complex) sorted in descending order.
        """
        coefficients = problem
        
        # For small polynomials, use direct methods
        n = len(coefficients) - 1
        
        if n == 1:
            # Linear: ax + b = 0 -> x = -b/a
            if CYTHON_AVAILABLE:
                return solve_linear(coefficients[0], coefficients[1])
            else:
                return [-complex(coefficients[1] / coefficients[0])]
        elif n == 2:
            # Quadratic: ax^2 + bx + c = 0
            a, b, c = coefficients
            if CYTHON_AVAILABLE:
                roots = solve_quadratic(a, b, c)
            else:
                discriminant = b**2 - 4*a*c
                sqrt_disc = np.sqrt(discriminant + 0j)
                root1 = (-b + sqrt_disc) / (2*a)
                root2 = (-b - sqrt_disc) / (2*a)
                roots = [root1, root2]
        else:
            # For larger polynomials, use final optimized methods
            roots = find_roots_final_optimized(coefficients)
        
        # Sort using the fastest method
        if CYTHON_AVAILABLE:
            roots = sort_roots(roots)
        elif NUMBA_AVAILABLE:
            roots_real = np.array([r.real for r in roots])
            roots_imag = np.array([r.imag for r in roots])
            roots_real, roots_imag = sort_roots_numba(roots_real, roots_imag)
            roots = [complex(roots_real[i], roots_imag[i]) for i in range(len(roots_real))]
        else:
            roots.sort(key=lambda z: (z.real, z.imag), reverse=True)
        return roots