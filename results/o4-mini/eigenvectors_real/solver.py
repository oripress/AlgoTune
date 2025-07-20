import numpy as np
import math
from scipy.linalg.lapack import get_lapack_funcs

class Solver:
    def solve(self, problem, **kwargs):
        n = problem.shape[0]
        # Trivial 1×1
        if n == 1:
            v00 = float(problem[0, 0])
            return [v00], [[1.0]]
        # Closed-form 2×2 symmetric
        if n == 2:
            a = problem[0, 0]; b = problem[0, 1]; c = problem[1, 1]
            tr = a + c
            det = a * c - b * b
            mid = 0.5 * tr
            rad = math.sqrt(max(mid * mid - det, 0.0))
            l1 = mid + rad
            l2 = mid - rad
            if abs(b) > 1e-16:
                v1 = [b, l1 - a]
                v2 = [b, l2 - a]
            else:
                if abs(a - l1) > abs(c - l1):
                    v1 = [l1 - c, b]
                    v2 = [l2 - c, b]
                else:
                    v1 = [1.0, 0.0]
                    v2 = [0.0, 1.0]
            n1 = math.hypot(v1[0], v1[1])
            if n1 != 0.0:
                v1 = [v1[0] / n1, v1[1] / n1]
            else:
                v1 = [1.0, 0.0]
            n2 = math.hypot(v2[0], v2[1])
            if n2 != 0.0:
                v2 = [v2[0] / n2, v2[1] / n2]
            else:
                v2 = [-v1[1], v1[0]]
            if l1 >= l2:
                return [l1, l2], [v1, v2]
            else:
                return [l2, l1], [v2, v1]
        # General case: LAPACK divide-and-conquer
        A = np.array(problem, dtype=np.double, order='F', copy=False)
        syevd, = get_lapack_funcs(('syevd',), (A,))
        w, v, info = syevd(A, lower=1, overwrite_a=1)
        # Reverse to descending order
        w = w[::-1]
        v = v[:, ::-1]
        # Return eigenvalues and eigenvectors (rows)
        return w.tolist(), v.T.tolist()