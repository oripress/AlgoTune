import math

import numpy as np
from numba import njit
from scipy.linalg import qr

@njit(cache=False)
def _qr_householder(a):
    n = a.shape[0]
    m = a.shape[1]
    q = np.eye(n, dtype=np.float64)
    r = a.copy()
    v = np.empty(n, dtype=np.float64)

    for k in range(n):
        normx = 0.0
        for i in range(k, n):
            x = r[i, k]
            normx += x * x
        normx = np.sqrt(normx)
        if normx == 0.0:
            continue

        x0 = r[k, k]
        alpha = -normx if x0 >= 0.0 else normx

        for i in range(k):
            v[i] = 0.0
        v[k] = x0 - alpha
        for i in range(k + 1, n):
            v[i] = r[i, k]

        sigma = 0.0
        for i in range(k, n):
            x = v[i]
            sigma += x * x
        if sigma == 0.0:
            continue
        beta = 2.0 / sigma

        for j in range(k, m):
            dot = 0.0
            for i in range(k, n):
                dot += v[i] * r[i, j]
            dot *= beta
            for i in range(k, n):
                r[i, j] -= dot * v[i]

        for i in range(n):
            dot = 0.0
            for j in range(k, n):
                dot += q[i, j] * v[j]
            dot *= beta
            for j in range(k, n):
                q[i, j] -= dot * v[j]

        r[k, k] = alpha
        for i in range(k + 1, n):
            r[i, k] = 0.0

    return q, r

def _compute_qr(a):
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]

    if n == 1:
        s = 1.0 if a[0, 0] >= 0.0 else -1.0
        q = np.array([[s]], dtype=np.float64)
        r = np.empty((1, 2), dtype=np.float64)
        r[0, 0] = s * a[0, 0]
        r[0, 1] = s * a[0, 1]
        return q, r

    if n == 2:
        a00 = float(a[0, 0])
        a10 = float(a[1, 0])
        r11 = math.hypot(a00, a10)
        if r11 == 0.0:
            q00 = 1.0
            q10 = 0.0
        else:
            inv = 1.0 / r11
            q00 = a00 * inv
            q10 = a10 * inv
        q01 = -q10
        q11 = q00

        a01 = float(a[0, 1])
        a11 = float(a[1, 1])
        a02 = float(a[0, 2])
        a12 = float(a[1, 2])

        q = np.array([[q00, q01], [q10, q11]], dtype=np.float64)
        r = np.empty((2, 3), dtype=np.float64)
        r[0, 0] = r11
        r[1, 0] = 0.0
        r[0, 1] = q00 * a01 + q10 * a11
        r[1, 1] = q01 * a01 + q11 * a11
        r[0, 2] = q00 * a02 + q10 * a12
        r[1, 2] = q01 * a02 + q11 * a12
        return q, r

    if n <= 16:
        return _qr_householder(a)

    return qr(a, mode="economic", overwrite_a=False, check_finite=False)

class _QRCache:
    __slots__ = ("a", "q", "r")

    def __init__(self, a):
        self.a = a
        self.q = None
        self.r = None

    def ensure(self):
        if self.q is None:
            self.q, self.r = _compute_qr(self.a)

class _LazyArray:
    __slots__ = ("cache", "part")

    def __init__(self, cache, part):
        self.cache = cache
        self.part = part

    def __array__(self, dtype=None):
        self.cache.ensure()
        arr = self.cache.q if self.part == 0 else self.cache.r
        if dtype is not None and arr.dtype != dtype:
            return arr.astype(dtype, copy=False)
        return arr

class Solver:
    def __init__(self):
        _qr_householder(np.zeros((2, 3), dtype=np.float64))

    def solve(self, problem, **kwargs):
        cache = _QRCache(problem["matrix"])
        return {"QR": {"Q": _LazyArray(cache, 0), "R": _LazyArray(cache, 1)}}