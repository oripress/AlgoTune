import cmath
import numpy as np

_OMEGA = complex(-0.5, 0.86602540378443864676372317075294)
_OMEGA2 = complex(-0.5, -0.86602540378443864676372317075294)

def _cbrt(z):
    if isinstance(z, complex):
        if z.imag == 0.0:
            x = z.real
            if x >= 0.0:
                return x ** (1.0 / 3.0)
            return -((-x) ** (1.0 / 3.0))
        return z ** (1.0 / 3.0)
    if z >= 0.0:
        return z ** (1.0 / 3.0)
    return -((-z) ** (1.0 / 3.0))

def _sort_small(roots):
    roots = list(roots)
    roots.sort(key=lambda z: (z.real, z.imag), reverse=True)
    return roots

def _sort_small_with_zeros(roots, trailing):
    roots = list(roots)
    if trailing:
        roots.extend(0.0 for _ in range(trailing))
    roots.sort(key=lambda z: (z.real, z.imag), reverse=True)
    return roots

def _cubic_roots_monic(b, c, d):
    bb = b * b
    p = c - bb / 3.0
    q = (2.0 * bb * b) / 27.0 - (b * c) / 3.0 + d
    half_q = 0.5 * q
    disc = half_q * half_q + (p * p * p) / 27.0
    shift = b / 3.0

    if disc == 0.0 and p == 0.0 and q == 0.0:
        r = -shift
        return r, r, r

    sqrt_disc = cmath.sqrt(disc)
    u = _cbrt(-half_q + sqrt_disc)
    if u == 0:
        v = _cbrt(-half_q - sqrt_disc)
    else:
        v = -p / (3.0 * u)

    return (
        u + v - shift,
        u * _OMEGA + v * _OMEGA2 - shift,
        u * _OMEGA2 + v * _OMEGA - shift,
    )

def _residual_ok(coeffs, roots, tol):
    arr = np.asarray(roots, dtype=np.complex128)
    vals = np.polyval(coeffs, arr)
    scale = np.sum(np.abs(coeffs))
    if scale == 0.0:
        return True
    bound = scale * max(1.0, float(np.max(np.abs(arr)))) ** (len(coeffs) - 1)
    return float(np.max(np.abs(vals))) <= tol * (bound + 1e-300)

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)

        start = 0
        while start < n and problem[start] == 0:
            start += 1
        if start == n:
            return []

        end = n - 1
        while problem[end] == 0:
            end -= 1
        trailing = n - 1 - end

        m = end - start
        if m == 0:
            if trailing == 0:
                return []
            return [0.0] * trailing

        coeffs = problem[start : end + 1]

        if m == 1:
            root = -coeffs[1] / coeffs[0]
            if trailing == 0:
                return [root]
            out = [root]
            out.extend(0.0 for _ in range(trailing))
            out.sort(key=lambda z: (z.real, z.imag), reverse=True)
            return out

        if m == 2:
            a = coeffs[0]
            b = coeffs[1] / a
            c = coeffs[2] / a
            disc = cmath.sqrt(b * b - 4.0 * c)
            if b >= 0.0:
                q = -0.5 * (b + disc)
            else:
                q = -0.5 * (b - disc)
            if q == 0:
                r1 = -0.5 * b
                r2 = r1
            else:
                r1 = q
                r2 = c / q
            return _sort_small_with_zeros((r1, r2), trailing)

        if m == 3:
            a = coeffs[0]
            b = coeffs[1] / a
            c = coeffs[2] / a
            d = coeffs[3] / a

            if (
                np.isfinite(b)
                and np.isfinite(c)
                and np.isfinite(d)
                and abs(b) <= 1e100
                and abs(c) <= 1e100
                and abs(d) <= 1e100
            ):
                roots = _cubic_roots_monic(b, c, d)
                if _residual_ok(coeffs, roots, 1e-9):
                    return _sort_small_with_zeros(roots, trailing)

        if m == 4:
            a = coeffs[0]
            b = coeffs[1] / a
            c = coeffs[2] / a
            d = coeffs[3] / a
            e = coeffs[4] / a

            if (
                np.isfinite(b)
                and np.isfinite(c)
                and np.isfinite(d)
                and np.isfinite(e)
                and abs(b) <= 1e75
                and abs(c) <= 1e75
                and abs(d) <= 1e75
                and abs(e) <= 1e75
            ):
                bb = b * b
                p = c - 0.375 * bb
                q = 0.125 * bb * b - 0.5 * b * c + d
                r = e - 0.01171875 * bb * bb + 0.0625 * bb * c - 0.25 * b * d
                shift = 0.25 * b

                if abs(q) <= 1e-14 * (1.0 + abs(p) + abs(r)):
                    u = cmath.sqrt(p * p - 4.0 * r)
                    y1 = cmath.sqrt(0.5 * (-p + u))
                    y2 = cmath.sqrt(0.5 * (-p - u))
                    roots = (
                        y1 - shift,
                        -y1 - shift,
                        y2 - shift,
                        -y2 - shift,
                    )
                else:
                    z1, z2, z3 = _cubic_roots_monic(2.0 * p, p * p - 4.0 * r, -(q * q))
                    z = z1
                    az = abs(z)
                    az2 = abs(z2)
                    if az2 > az:
                        z = z2
                        az = az2
                    if abs(z3) > az:
                        z = z3
                    s = cmath.sqrt(z)
                    if abs(s) > 1e-15:
                        t1 = cmath.sqrt(-2.0 * p - z + 2.0 * q / s)
                        t2 = cmath.sqrt(-2.0 * p - z - 2.0 * q / s)
                        roots = (
                            0.5 * (-s + t1) - shift,
                            0.5 * (-s - t1) - shift,
                            0.5 * (s + t2) - shift,
                            0.5 * (s - t2) - shift,
                        )
                    else:
                        roots = None

                if roots is not None and _residual_ok(coeffs, roots, 1e-8):
                    return _sort_small_with_zeros(roots, trailing)

        roots = np.roots(coeffs)
        if trailing:
            out = np.empty(m + trailing, dtype=np.complex128)
            out[:m] = roots
            out[m:] = 0.0
            return np.sort_complex(out)[::-1]
        return np.sort_complex(roots)[::-1]