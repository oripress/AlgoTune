import cmath
import math
from typing import Any

import numpy as np

class Solver:
    def __init__(self) -> None:
        self._asarray = np.asarray
        self._eigvals = np.linalg.eigvals
        self._lexsort = np.lexsort
        self._ndarray = np.ndarray
        self._omega = complex(-0.5, 0.8660254037844386)
        self._omega2 = complex(-0.5, -0.8660254037844386)

    @staticmethod
    def _cleanup(z: complex) -> complex:
        if -1e-12 < z.imag < 1e-12:
            return complex(z.real, 0.0)
        return z

    @staticmethod
    def _sort3(z1: complex, z2: complex, z3: complex) -> list[complex]:
        if (z1.real < z2.real) or (z1.real == z2.real and z1.imag < z2.imag):
            z1, z2 = z2, z1
        if (z2.real < z3.real) or (z2.real == z3.real and z2.imag < z3.imag):
            z2, z3 = z3, z2
        if (z1.real < z2.real) or (z1.real == z2.real and z1.imag < z2.imag):
            z1, z2 = z2, z1
        return [z1, z2, z3]

    def solve(self, problem, **kwargs) -> Any:
        p = problem
        n = len(p)

        if n == 0:
            return []

        row0 = p[0]

        if n == 1:
            return [complex(row0[0])]

        if n == 2:
            a00 = row0[0]
            a01 = row0[1]
            row1 = p[1]
            a10 = row1[0]
            a11 = row1[1]
            tr = a00 + a11
            disc = (a00 - a11) * (a00 - a11) + 4.0 * a01 * a10
            if disc >= 0.0:
                half_disc = 0.5 * math.sqrt(disc)
            else:
                half_disc = 0.5j * math.sqrt(-disc)
            half_tr = 0.5 * tr
            z1 = complex(half_tr + half_disc)
            z2 = complex(half_tr - half_disc)
            if (z1.real < z2.real) or (z1.real == z2.real and z1.imag < z2.imag):
                z1, z2 = z2, z1
            return [z1, z2]

        if n == 3:
            a = row0[0]
            b = row0[1]
            c = row0[2]
            row1 = p[1]
            d = row1[0]
            e = row1[1]
            f = row1[2]
            row2 = p[2]
            g = row2[0]
            h = row2[1]
            i = row2[2]

            tr = a + e + i
            s = a * e + a * i + e * i - b * d - c * g - f * h
            det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

            pc = s - tr * tr / 3.0
            qc = tr * s / 3.0 - det - 2.0 * tr * tr * tr / 27.0
            delta = 0.25 * qc * qc + pc * pc * pc / 27.0
            sqrt_delta = cmath.sqrt(delta)

            u = (-0.5 * qc + sqrt_delta) ** (1.0 / 3.0)
            v = (-0.5 * qc - sqrt_delta) ** (1.0 / 3.0)
            if abs(u) > 1e-15:
                v = -pc / (3.0 * u)
            elif abs(v) > 1e-15:
                u = -pc / (3.0 * v)

            shift = tr / 3.0
            z1 = self._cleanup(u + v + shift)
            z2 = self._cleanup(self._omega * u + self._omega2 * v + shift)
            z3 = self._cleanup(self._omega2 * u + self._omega * v + shift)
            return self._sort3(z1, z2, z3)

        vals = self._eigvals(p)
        idx = self._lexsort((-vals.imag, -vals.real))
        return vals[idx].tolist()