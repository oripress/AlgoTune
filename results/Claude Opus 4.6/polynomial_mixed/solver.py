import numpy as np
from scipy.linalg.lapack import dgebal, dhseqr
import cmath

def _cubic_roots(a, b, c, d):
    """Solve ax^3 + bx^2 + cx + d = 0"""
    # Convert to depressed cubic t^3 + pt + q = 0
    # x = t - b/(3a)
    a_inv = 1.0 / a
    b1 = b * a_inv
    c1 = c * a_inv
    d1 = d * a_inv
    
    p = c1 - b1*b1/3.0
    q = 2.0*b1*b1*b1/27.0 - b1*c1/3.0 + d1
    
    disc = -4.0*p*p*p - 27.0*q*q
    shift = -b1/3.0
    
    if disc > 0:
        # Three real roots
        m = 2.0 * (-p/3.0)**0.5
        theta = cmath.acos(3.0*q/(p*m)).real / 3.0
        import math
        r1 = complex(m * math.cos(theta) + shift)
        r2 = complex(m * math.cos(theta - 2.0*math.pi/3.0) + shift)
        r3 = complex(m * math.cos(theta - 4.0*math.pi/3.0) + shift)
        return [r1, r2, r3]
    else:
        # One real root, two complex conjugate roots
        # Use Cardano's formula
        D = q*q/4.0 + p*p*p/27.0
        if D >= 0:
            sqrtD = D**0.5
            u = (-q/2.0 + sqrtD)
            v = (-q/2.0 - sqrtD)
            u = u**(1./3.) if u >= 0 else -((-u)**(1./3.))
            v = v**(1./3.) if v >= 0 else -((-v)**(1./3.))
            r1 = complex(u + v + shift)
            re2 = -(u+v)/2.0 + shift
            im2 = (u-v) * 3.0**0.5 / 2.0
            r2 = complex(re2, im2)
            r3 = complex(re2, -im2)
            return [r1, r2, r3]
        else:
            sqrtD = (-D)**0.5
            u_c = complex(-q/2.0, sqrtD)
            # cube root of complex number
            r_abs = abs(u_c)**(1./3.)
            theta = cmath.phase(u_c) / 3.0
            import math
            u = r_abs * complex(math.cos(theta), math.sin(theta))
            v = complex(p / (3.0 * u.real * 2), 0) if abs(u) > 1e-15 else complex(0)
            # Actually just fallback to numpy for edge cases
            return None


class Solver:
    def solve(self, problem, **kwargs):
        coeffs = np.asarray(problem, dtype=np.float64)
        
        # Strip leading zeros
        non_zero = np.flatnonzero(coeffs)
        nz_len = len(non_zero)
        if nz_len == 0:
            return []
        start = non_zero[0]
        end = non_zero[-1]
        
        if start > 0:
            coeffs = coeffs[start:]
            end -= start
        
        n = len(coeffs) - 1
        if n == 0:
            return []
        if n == 1:
            return [complex(-coeffs[1] / coeffs[0])]
        
        # Trailing zeros = zero roots
        trailing = n - end
        
        if trailing > 0:
            coeffs_r = coeffs[:end + 1]
            nr = end
        else:
            coeffs_r = coeffs
            nr = n
        
        if nr == 0:
            return [0j] * n
        
        if nr == 1:
            r = complex(-coeffs_r[1] / coeffs_r[0])
            if trailing == 0:
                return [r]
            if r.real >= 0:
                return [r] + [0j] * trailing
            return [0j] * trailing + [r]
        
        if nr == 2:
            a, b, c = float(coeffs_r[0]), float(coeffs_r[1]), float(coeffs_r[2])
            disc = b*b - 4*a*c
            two_a = 2*a
            if disc >= 0:
                sq = disc**0.5
                r1 = complex((-b + sq) / two_a)
                r2 = complex((-b - sq) / two_a)
            else:
                sq = (-disc)**0.5
                re = -b / two_a
                im = sq / two_a
                r1 = complex(re, im)
                r2 = complex(re, -im)
            roots = [r1, r2] + [0j] * trailing
            return sorted(roots, key=lambda z: (z.real, z.imag), reverse=True)
        
        # General case: companion matrix eigenvalue approach
        # Build companion matrix (already upper Hessenberg form)
        A = np.zeros((nr, nr), dtype=np.float64, order='F')
        A[0, :] = -coeffs_r[1:] / coeffs_r[0]
        np.fill_diagonal(A[1:, :], 1.0)
        
        # Balance + Hessenberg QR
        ba, lo, hi, scale, info = dgebal('B', A, overwrite_a=1)
        result = dhseqr(ba, compute_q=0, overwrite_a=1)
        wr, wi = result[0], result[1]
        
        # Build complex roots and sort
        roots_c = wr + 1j * wi
        
        if trailing > 0:
            total = nr + trailing
            all_roots = np.empty(total, dtype=np.complex128)
            all_roots[:nr] = roots_c
            all_roots[nr:] = 0.0
            roots_c = all_roots
        
        order = np.lexsort((roots_c.imag, roots_c.real))[::-1]
        return roots_c[order].tolist()