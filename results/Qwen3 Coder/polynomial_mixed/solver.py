from typing import List, Any
import numpy as np

class Solver:
    def solve(self, problem: List[float], **kwargs) -> List[complex]:
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
        # Handle special cases without numpy conversion for small inputs
        if not problem:
            return []

        # Handle edge cases more efficiently
        if not problem or all(c == 0 for c in problem):
            return []
            
        # Remove leading zeros efficiently
        first_nonzero = 0
        while first_nonzero < len(problem) and problem[first_nonzero] == 0:
            first_nonzero += 1
            
        coefficients = problem[first_nonzero:]
        
        # Handle special cases
        n = len(coefficients)
        if n <= 1:
            # Constant polynomial has no roots unless it's zero (already handled above)
            return []
            
        # For linear polynomial: ax + b = 0 => x = -b/a
        if n == 2:
            a, b = coefficients[0], coefficients[1]
            return [-b/a]
            
        # For quadratic polynomials, use the quadratic formula for better accuracy
        if n == 3:
            a, b, c = coefficients[0], coefficients[1], coefficients[2]
            # Use a more efficient approach for the quadratic formula
            discriminant = b*b - 4*a*c
            if discriminant >= 0:
                # Real roots
                sqrt_discriminant = np.sqrt(discriminant)
                root1 = (-b + sqrt_discriminant) / (2*a)
                root2 = (-b - sqrt_discriminant) / (2*a)
            else:
                # Complex roots
                sqrt_discriminant = np.sqrt(-discriminant) * 1j
                root1 = (-b + sqrt_discriminant) / (2*a)
                root2 = (-b - sqrt_discriminant) / (2*a)
            # Sort manually for just 2 elements
            if (root1.real > root2.real) or (root1.real == root2.real and root1.imag >= root2.imag):
                return [root1, root2]
            else:
                return [root2, root1]
                return [root2, root1]
        
        # For higher degree polynomials, use numpy's roots function directly
        computed_roots = np.roots(coefficients)
        
        # Optimize sorting based on array size
        if len(computed_roots) <= 10:
            # For small arrays, use a more direct sorting approach
            root_data = [(r.real, r.imag) for r in computed_roots]
            sorted_indices = sorted(range(len(root_data)), key=lambda i: (-root_data[i][0], -root_data[i][1]))
            return [computed_roots[i] for i in sorted_indices]
        else:
            # For larger arrays, use numpy's lexsort
            sorted_indices = np.lexsort((-computed_roots.imag, -computed_roots.real))
            return computed_roots[sorted_indices].tolist()