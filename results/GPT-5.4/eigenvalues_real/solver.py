from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np
from scipy import linalg

class Solver:
    def __init__(self) -> None:
        self._ndarray = np.ndarray
        self._pi_2_over_3 = 2.0 * math.pi / 3.0
        self._dense_solver = self._select_dense_solver()
        self._tridiag_solver = self._select_tridiagonal_solver()

    def _select_dense_solver(self) -> Callable[[np.ndarray], np.ndarray]:
        candidates: list[Callable[[np.ndarray], np.ndarray]] = []

        def numpy_solver(a: np.ndarray) -> np.ndarray:
            return np.linalg.eigvalsh(a, UPLO="L")

        candidates.append(numpy_solver)

        def scipy_solver(a: np.ndarray) -> np.ndarray:
            return linalg.eigvalsh(
                a,
                lower=True,
                overwrite_a=False,
                check_finite=False,
            )

        candidates.append(scipy_solver)

        lapack_mod = linalg.lapack

        try:
            dsyevd = getattr(lapack_mod, "dsyevd", None)
            if dsyevd is not None:

                def lapack_dsyevd(a: np.ndarray, _f=dsyevd) -> np.ndarray:
                    w, _v, info = _f(a, compute_v=0, lower=1, overwrite_a=0)
                    if info != 0:
                        raise np.linalg.LinAlgError(info)
                    return w

                probe = np.array([[1.0, 0.25], [0.25, 2.0]], dtype=np.float64, order="F")
                out = lapack_dsyevd(probe)
                if out.shape == (2,) and np.all(np.isfinite(out)):
                    candidates.append(lapack_dsyevd)
        except Exception:
            pass

        try:
            dsyevr = getattr(lapack_mod, "dsyevr", None)
            if dsyevr is not None:

                def lapack_dsyevr(a: np.ndarray, _f=dsyevr) -> np.ndarray:
                    w, _z, m, _isuppz, info = _f(
                        a,
                        compute_v=0,
                        range="A",
                        lower=1,
                        overwrite_a=0,
                    )
                    if info != 0 or m != a.shape[0]:
                        raise np.linalg.LinAlgError(info)
                    return w

                probe = np.array([[1.0, 0.25], [0.25, 2.0]], dtype=np.float64, order="F")
                out = lapack_dsyevr(probe)
                if out.shape == (2,) and np.all(np.isfinite(out)):
                    candidates.append(lapack_dsyevr)
        except Exception:
            pass

        rng = np.random.default_rng(0)
        tests = []
        for size in (8, 32, 64):
            g = rng.standard_normal((size, size))
            tests.append(np.array((g + g.T) * 0.5, dtype=np.float64, order="F"))

        best = candidates[0]
        best_time = float("inf")
        for cand in candidates:
            try:
                total = 0.0
                for test in tests:
                    warm = cand(test)
                    if warm.shape != (test.shape[0],) or not np.all(np.isfinite(warm)):
                        raise ValueError("bad output")
                    t0 = time.perf_counter()
                    cand(test)
                    total += time.perf_counter() - t0
                if total < best_time:
                    best_time = total
                    best = cand
            except Exception:
                continue
        return best

    def _select_tridiagonal_solver(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        drivers = []
        d0 = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        e0 = np.array([-1.0, -1.0], dtype=np.float64)

        for driver in ("auto", "stemr", "sterf"):
            try:
                out = linalg.eigvalsh_tridiagonal(
                    d0,
                    e0,
                    check_finite=False,
                    lapack_driver=driver,
                )
                if out.shape == (3,) and np.all(np.isfinite(out)):
                    drivers.append(driver)
            except Exception:
                continue

        if not drivers:
            return lambda d, e: linalg.eigvalsh_tridiagonal(d, e, check_finite=False)

        rng = np.random.default_rng(1)
        d = rng.standard_normal(96)
        e = rng.standard_normal(95)

        best_driver = drivers[0]
        best_time = float("inf")
        for driver in drivers:
            try:
                linalg.eigvalsh_tridiagonal(
                    d,
                    e,
                    check_finite=False,
                    lapack_driver=driver,
                )
                t0 = time.perf_counter()
                linalg.eigvalsh_tridiagonal(
                    d,
                    e,
                    check_finite=False,
                    lapack_driver=driver,
                )
                elapsed = time.perf_counter() - t0
                if elapsed < best_time:
                    best_time = elapsed
                    best_driver = driver
            except Exception:
                continue

        return lambda d, e, _driver=best_driver: linalg.eigvalsh_tridiagonal(
            d,
            e,
            check_finite=False,
            lapack_driver=_driver,
        )

    def solve(self, problem, **kwargs) -> Any:
        if isinstance(problem, self._ndarray):
            a = problem
            n = a.shape[0]
        else:
            a = None
            n = len(problem)

        if n == 0:
            return []

        if n == 1:
            return [float(a[0, 0] if a is not None else problem[0][0])]

        if n == 2:
            if a is None:
                x = float(problem[0][0])
                y = float(problem[0][1])
                z = float(problem[1][1])
            else:
                x = float(a[0, 0])
                y = float(a[0, 1])
                z = float(a[1, 1])
            t = math.hypot(x - z, 2.0 * y)
            s = x + z
            return [0.5 * (s + t), 0.5 * (s - t)]

        if n == 3:
            if a is None:
                a00 = float(problem[0][0])
                a01 = float(problem[0][1])
                a02 = float(problem[0][2])
                a11 = float(problem[1][1])
                a12 = float(problem[1][2])
                a22 = float(problem[2][2])
            else:
                a00 = float(a[0, 0])
                a01 = float(a[0, 1])
                a02 = float(a[0, 2])
                a11 = float(a[1, 1])
                a12 = float(a[1, 2])
                a22 = float(a[2, 2])

            p1 = a01 * a01 + a02 * a02 + a12 * a12
            if p1 == 0.0:
                if a00 < a11:
                    a00, a11 = a11, a00
                if a11 < a22:
                    a11, a22 = a22, a11
                if a00 < a11:
                    a00, a11 = a11, a00
                return [a00, a11, a22]

            q = (a00 + a11 + a22) / 3.0
            b00 = a00 - q
            b11 = a11 - q
            b22 = a22 - q
            p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * p1

            if p2 > 0.0:
                p = math.sqrt(p2 / 6.0)
                inv_p = 1.0 / p
                c00 = b00 * inv_p
                c01 = a01 * inv_p
                c02 = a02 * inv_p
                c11 = b11 * inv_p
                c12 = a12 * inv_p
                c22 = b22 * inv_p

                det_c = (
                    c00 * (c11 * c22 - c12 * c12)
                    - c01 * (c01 * c22 - c12 * c02)
                    + c02 * (c01 * c12 - c11 * c02)
                )
                r = 0.5 * det_c
                if r <= -1.0:
                    phi = math.pi / 3.0
                elif r >= 1.0:
                    phi = 0.0
                else:
                    phi = math.acos(r) / 3.0

                two_p = 2.0 * p
                eig1 = q + two_p * math.cos(phi)
                eig3 = q + two_p * math.cos(phi + self._pi_2_over_3)
                eig2 = 3.0 * q - eig1 - eig3

                if math.isfinite(eig1) and math.isfinite(eig2) and math.isfinite(eig3):
                    if eig1 < eig2:
                        eig1, eig2 = eig2, eig1
                    if eig2 < eig3:
                        eig2, eig3 = eig3, eig2
                    if eig1 < eig2:
                        eig1, eig2 = eig2, eig1
                    return [eig1, eig2, eig3]

        if a is None:
            a = np.array(problem, dtype=np.float64, order="F", copy=False)
        elif a.dtype != np.float64 or not a.flags.f_contiguous:
            a = np.array(a, dtype=np.float64, order="F", copy=False)

        if n >= 6:
            diag = a.diagonal
            is_tridiagonal = True
            for k in range(2, n):
                if np.any(diag(k)):
                    is_tridiagonal = False
                    break
            if is_tridiagonal:
                vals = self._tridiag_solver(diag(0), diag(1))
                return vals[::-1].tolist()

        vals = self._dense_solver(a)
        return vals[::-1].tolist()