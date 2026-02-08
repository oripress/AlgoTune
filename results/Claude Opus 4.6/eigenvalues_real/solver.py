import numpy as np
import math

class Solver:
    def solve(self, problem, **kwargs) -> list:
        """Compute eigenvalues of a symmetric matrix in descending order."""
        if isinstance(problem, np.ndarray):
            if problem.dtype == np.float64:
                A = problem
            else:
                A = problem.astype(np.float64)
        else:
            A = np.array(problem, dtype=np.float64)
        
        n = A.shape[0]
        
        if n == 0:
            return []
        if n == 1:
            return [float(A[0, 0])]
        if n == 2:
            a, b, d = A[0, 0], A[0, 1], A[1, 1]
            trace = a + d
            disc = (a - d) * (a - d) + 4.0 * b * b
            sqrt_disc = math.sqrt(disc)
            return [(trace + sqrt_disc) * 0.5, (trace - sqrt_disc) * 0.5]
        if n == 3:
            a00, a01, a02 = A[0, 0], A[0, 1], A[0, 2]
            a11, a12 = A[1, 1], A[1, 2]
            a22 = A[2, 2]
            
            p1 = a01*a01 + a02*a02 + a12*a12
            q = (a00 + a11 + a22) / 3.0
            p2 = (a00 - q)**2 + (a11 - q)**2 + (a22 - q)**2 + 2.0*p1
            p = math.sqrt(p2 / 6.0)
            
            if p < 1e-15:
                return [q, q, q]
            
            inv_p = 1.0 / p
            B00 = inv_p * (a00 - q)
            B11 = inv_p * (a11 - q)
            B22 = inv_p * (a22 - q)
            B01 = inv_p * a01
            B02 = inv_p * a02
            B12 = inv_p * a12
            
            det_B = B00*(B11*B22 - B12*B12) - B01*(B01*B22 - B12*B02) + B02*(B01*B12 - B11*B02)
            r = det_B * 0.5
            r = max(-1.0, min(1.0, r))
            phi = math.acos(r) / 3.0
            
            eig1 = q + 2.0*p*math.cos(phi)
            eig3 = q + 2.0*p*math.cos(phi + 2.0943951023931953)
            eig2 = 3.0*q - eig1 - eig3
            
            result = [eig1, eig2, eig3]
            result.sort(reverse=True)
            return result
        
        # numpy.linalg.eigvalsh uses LAPACK dsyevd internally
        eigenvalues = np.linalg.eigvalsh(A)
        return eigenvalues[::-1].tolist()