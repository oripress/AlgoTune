import numpy as np
import scipy.linalg
from numba import njit
import math
import cmath
@njit
def solve_quadratic(coeffs):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    delta = b*b - 4*a*c
    if delta >= 0:
        sqrt_delta = math.sqrt(delta)
        r1 = (-b + sqrt_delta) / (2*a)
        r2 = (-b - sqrt_delta) / (2*a)
        return [complex(r1, 0.0), complex(r2, 0.0)]
    else:
        sqrt_delta = math.sqrt(-delta)
        real = -b / (2*a)
        imag = sqrt_delta / (2*a)
        return [complex(real, imag), complex(real, -imag)]

@njit
def solve_cubic(coeffs):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    d = coeffs[3]
    
    inv_a = 1.0 / a
    A = b * inv_a
    B = c * inv_a
    C = d * inv_a
    
    sq_A = A * A
    p = B - sq_A / 3.0
    q = (2.0 * sq_A * A) / 27.0 - (A * B) / 3.0 + C
    
    half_q = q * 0.5
    third_p = p / 3.0
    D = half_q * half_q + third_p * third_p * third_p
    
    offset = A / 3.0
    
    if abs(D) < 1e-14:
        if abs(q) < 1e-14:
             return [complex(-offset, 0.0)] * 3
        else:
            u = -math.pow(abs(half_q), 1.0/3.0) if half_q > 0 else math.pow(abs(half_q), 1.0/3.0)
            y1 = 2.0 * u
            y2 = -u
            return [
                complex(y1 - offset, 0.0),
                complex(y2 - offset, 0.0),
                complex(y2 - offset, 0.0)
            ]
    elif D > 0:
        sqrt_D = math.sqrt(D)
        u_cubed = -half_q + sqrt_D
        v_cubed = -half_q - sqrt_D
        
        u = math.pow(u_cubed, 1.0/3.0) if u_cubed >= 0 else -math.pow(-u_cubed, 1.0/3.0)
        v = math.pow(v_cubed, 1.0/3.0) if v_cubed >= 0 else -math.pow(-v_cubed, 1.0/3.0)
        
        y1 = u + v
        real_part = -0.5 * (u + v)
        imag_part = 0.8660254037844386 * (u - v)
        
        return [
            complex(y1 - offset, 0.0),
            complex(real_part - offset, imag_part),
            complex(real_part - offset, -imag_part)
        ]
    else:
        r = math.sqrt(-third_p)
        phi = math.acos(-half_q / (r * r * r))
        
        y1 = 2.0 * r * math.cos(phi / 3.0)
        y2 = 2.0 * r * math.cos((phi + 2.0 * math.pi) / 3.0)
        y3 = 2.0 * r * math.cos((phi + 4.0 * math.pi) / 3.0)
        
        return [
            complex(y1 - offset, 0.0),
            complex(y2 - offset, 0.0),
            complex(y3 - offset, 0.0)
        ]
class Solver:
    def solve(self, problem: list[float]) -> list[complex]:
        """
        Solve the polynomial problem by finding all roots of the polynomial.
        """
        # Filter leading zeros
        start_idx = 0
        n_coeffs = len(problem)
        while start_idx < n_coeffs and problem[start_idx] == 0:
            start_idx += 1
        
        if start_idx == n_coeffs:
            return []
            
        # Effective coefficients
        # We can slice, but for small lists it's fine.
        # If we slice, we create a new list.
        coeffs = problem[start_idx:]
        n = len(coeffs) - 1
        
        if n == 1:
            # ax + b = 0
            val = -coeffs[1] / coeffs[0]
            return [complex(val, 0.0)]
            
        if n == 2:
            c_arr = np.array(coeffs, dtype=np.float64)
            return solve_quadratic(c_arr)
            
        if n == 3:
            c_arr = np.array(coeffs, dtype=np.float64)
            return solve_cubic(c_arr)
        
        # For higher degrees, use np.roots
        # np.roots expects the full list including zeros if we pass it, 
        # but we already stripped them.
        computed_roots = np.roots(coeffs)
        
        # Sorting
        # We can optimize sorting by converting to a structure that is faster to sort?
        # Python's sort is Timsort, quite fast.
        # The key extraction might be slow.
        
        # Convert to list
        roots_list = computed_roots.tolist()
        roots_list.sort(key=lambda z: (z.real, z.imag), reverse=True)
        return roots_list