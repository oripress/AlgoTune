import cmath
import math

import numpy as np

class Solver:
    @staticmethod
    def _vec2(b, la, ld, c):
        n0 = b.real * b.real + b.imag * b.imag + la.real * la.real + la.imag * la.imag
        n1 = ld.real * ld.real + ld.imag * ld.imag + c.real * c.real + c.imag * c.imag
        if n0 >= n1:
            if n0 <= 0.0:
                return None
            inv = 1.0 / math.sqrt(n0)
            return [b * inv, la * inv]
        if n1 <= 0.0:
            return None
        inv = 1.0 / math.sqrt(n1)
        return [ld * inv, c * inv]

    def solve(self, problem, **kwargs):
        a = np.asarray(problem)
        n = a.shape[0]

        if n == 1:
            return [[1.0]]

        if n == 2:
            a00 = a[0, 0]
            a01 = a[0, 1]
            a10 = a[1, 0]
            a11 = a[1, 1]

            tr = a00 + a11
            diff = a00 - a11
            disc = diff * diff + 4.0 * a01 * a10
            root = cmath.sqrt(complex(disc))
            scale = abs(a00) + abs(a01) + abs(a10) + abs(a11) + 1.0

            if abs(root) > 1e-12 * scale:
                l0 = 0.5 * (tr + root)
                l1 = 0.5 * (tr - root)
                v0 = self._vec2(complex(a01), l0 - a00, l0 - a11, complex(a10))
                v1 = self._vec2(complex(a01), l1 - a00, l1 - a11, complex(a10))
                if v0 is not None and v1 is not None:
                    if (l0.real < l1.real) or (l0.real == l1.real and l0.imag < l1.imag):
                        return [v1, v0]
                    return [v0, v1]

        vals, vecs = np.linalg.eig(a)
        idx = np.lexsort((np.arange(n), -vals.imag, -vals.real))
        return vecs[:, idx].T.tolist()