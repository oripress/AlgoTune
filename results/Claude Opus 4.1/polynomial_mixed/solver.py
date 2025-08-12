import numpy as np
from numba import njit, complex128, float64
import numba as nb

class Solver:
    def __init__(self):
        # Pre-compile the functions
        _ = solve_quadratic(np.array([1.0, -2.0, 1.0]))
        _ = solve_cubic(np.array([1.0, 0.0, -1.0, 0.0]))
        _ = fast_sort(np.array([1+1j, 2+0j, 0+2j], dtype=np.complex128))
        
    def solve(self, problem: list[float]) -> list[complex]:
        """
        Solve the polynomial problem by finding all roots of the polynomial.
        """
        coefficients = np.asarray(problem, dtype=np.float64)
        n = len(coefficients) - 1  # degree of polynomial
        
        # Handle special cases for low-degree polynomials
        if n == 0:
            return []
        elif n == 1:
            # Linear: ax + b = 0
            return [complex(-coefficients[1] / coefficients[0])]
        elif n == 2:
            # Quadratic
            roots = solve_quadratic(coefficients)
            return fast_sort(roots).tolist()
        elif n == 3:
            # Cubic
            roots = solve_cubic(coefficients)
            return fast_sort(roots).tolist()
        else:
            # Higher degree - use numpy's optimized method
            roots = np.roots(coefficients)
            return fast_sort(roots).tolist()

@njit(complex128[:](complex128[:]), cache=True)
def fast_sort(roots):
    """Fast sorting of complex numbers by real then imaginary parts, descending"""
    n = len(roots)
    sorted_roots = roots.copy()
    
    # Simple insertion sort for small arrays, bubble sort for larger
    if n <= 10:
        for i in range(1, n):
            key = sorted_roots[i]
            j = i - 1
            while j >= 0 and (sorted_roots[j].real < key.real or 
                            (sorted_roots[j].real == key.real and sorted_roots[j].imag < key.imag)):
                sorted_roots[j + 1] = sorted_roots[j]
                j -= 1
            sorted_roots[j + 1] = key
    else:
        # Quicksort-style for larger arrays
        for i in range(n):
            for j in range(0, n-i-1):
                if (sorted_roots[j].real < sorted_roots[j+1].real or 
                   (sorted_roots[j].real == sorted_roots[j+1].real and 
                    sorted_roots[j].imag < sorted_roots[j+1].imag)):
                    sorted_roots[j], sorted_roots[j+1] = sorted_roots[j+1], sorted_roots[j]
    
    return sorted_roots

@njit(complex128(complex128), cache=True)
def complex_cbrt(z):
    """Compute complex cube root"""
    if z == 0:
        return complex(0, 0)
    # Convert to polar form
    r = abs(z)
    theta = np.angle(z)
    # Cube root in polar form
    r_cbrt = r ** (1/3)
    theta_cbrt = theta / 3
    # Convert back to rectangular
    return r_cbrt * (np.cos(theta_cbrt) + 1j * np.sin(theta_cbrt))

@njit(complex128[:](float64[:]), cache=True)
def solve_quadratic(coeffs):
    """Solve quadratic equation ax^2 + bx + c = 0 using stable formula"""
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    
    # Use numerically stable formula
    discriminant = b*b - 4*a*c
    
    if discriminant >= 0:
        # Real roots - use stable formula to avoid cancellation
        sqrt_disc = np.sqrt(discriminant)
        if b >= 0:
            t = -0.5 * (b + sqrt_disc)
        else:
            t = -0.5 * (b - sqrt_disc)
        
        roots = np.empty(2, dtype=np.complex128)
        roots[0] = complex(t / a, 0)
        roots[1] = complex(c / t, 0)
    else:
        # Complex roots
        sqrt_disc = np.sqrt(-discriminant)
        roots = np.empty(2, dtype=np.complex128)
        roots[0] = complex(-b / (2*a), sqrt_disc / (2*a))
        roots[1] = complex(-b / (2*a), -sqrt_disc / (2*a))
    
    return roots

@njit(complex128[:](float64[:]), cache=True)
def solve_cubic(coeffs):
    """Solve cubic equation using depressed cubic method"""
    a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    
    # Normalize
    b = b / a
    c = c / a
    d = d / a
    
    # Convert to depressed cubic t^3 + p*t + q = 0
    # using substitution x = t - b/3
    f = c - b*b/3
    g = (2*b*b*b - 9*b*c + 27*d) / 27
    h = g*g/4 + f*f*f/27
    
    roots = np.empty(3, dtype=np.complex128)
    
    if f == 0 and g == 0 and h == 0:
        # All roots are equal
        roots[0] = complex(-b/3, 0)
        roots[1] = roots[0]
        roots[2] = roots[0]
    elif h > 0:
        # One real root, two complex conjugate roots
        r = -(g/2) + np.sqrt(h)
        if r >= 0:
            s = r ** (1/3)
        else:
            s = -((-r) ** (1/3))
        
        t = -(g/2) - np.sqrt(h)
        if t >= 0:
            u = t ** (1/3)
        else:
            u = -((-t) ** (1/3))
        
        roots[0] = complex((s + u) - b/3, 0)
        roots[1] = complex(-(s + u)/2 - b/3, (s - u) * np.sqrt(3)/2)
        roots[2] = complex(-(s + u)/2 - b/3, -(s - u) * np.sqrt(3)/2)
    else:
        # Three real roots
        i = np.sqrt(((g*g)/4) - h)
        j = i ** (1/3)
        k = np.arccos(-g / (2*i))
        m = np.cos(k/3)
        n = np.sqrt(3) * np.sin(k/3)
        p = -b/3
        
        roots[0] = complex(2*j*m + p, 0)
        roots[1] = complex(-j*(m + n) + p, 0)
        roots[2] = complex(-j*(m - n) + p, 0)
    
    return roots