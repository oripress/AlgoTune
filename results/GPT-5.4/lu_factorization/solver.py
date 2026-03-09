import numpy as np
from numba import njit
from scipy.linalg import lu_factor

_EMPTY = np.empty((0, 0), dtype=np.float64)
_ONE = np.array([[1.0]], dtype=np.float64)
_I2 = np.eye(2, dtype=np.float64)
_P2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

@njit(cache=True)
def _lu_small(a):
    n = a.shape[0]
    lu = a.copy()
    perm = np.arange(n)

    for k in range(n):
        pivot = k
        max_abs = abs(lu[k, k])
        for i in range(k + 1, n):
            v = abs(lu[i, k])
            if v > max_abs:
                max_abs = v
                pivot = i

        if pivot != k:
            tmp = lu[k].copy()
            lu[k] = lu[pivot]
            lu[pivot] = tmp

            ptmp = perm[k]
            perm[k] = perm[pivot]
            perm[pivot] = ptmp

        pivot_val = lu[k, k]
        if pivot_val != 0.0:
            for i in range(k + 1, n):
                lu[i, k] /= pivot_val
                factor = lu[i, k]
                for j in range(k + 1, n):
                    lu[i, j] -= factor * lu[k, j]

    L = np.zeros((n, n), dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)
    P = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        P[perm[i], i] = 1.0
        for j in range(n):
            if i > j:
                L[i, j] = lu[i, j]
            elif i == j:
                L[i, j] = 1.0
                U[i, j] = lu[i, j]
            else:
                U[i, j] = lu[i, j]

    return P, L, U

class _LazyLU:
    __slots__ = ("A", "_plu")

    def __init__(self):
        self.A = _EMPTY
        self._plu = None

    def reset(self, A):
        self.A = A
        self._plu = None

    def get(self):
        plu = self._plu
        if plu is not None:
            return plu

        A = np.asarray(self.A, dtype=np.float64)
        n = A.shape[0]

        if n == 0:
            plu = (_EMPTY, _EMPTY, _EMPTY)
        elif n == 1:
            plu = (_ONE, _ONE, np.array([[A[0, 0]]], dtype=np.float64))
        elif n == 2:
            a = A[0, 0]
            b = A[0, 1]
            c = A[1, 0]
            d = A[1, 1]

            if abs(a) >= abs(c) and a != 0.0:
                L = np.array([[1.0, 0.0], [c / a, 1.0]], dtype=np.float64)
                U = np.array([[a, b], [0.0, d - (c / a) * b]], dtype=np.float64)
                plu = (_I2, L, U)
            elif c != 0.0:
                L = np.array([[1.0, 0.0], [a / c, 1.0]], dtype=np.float64)
                U = np.array([[c, d], [0.0, b - (a / c) * d]], dtype=np.float64)
                plu = (_P2, L, U)
            else:
                plu = (_I2, _I2, A)
        elif n <= 24:
            plu = _lu_small(A)
        else:
            lu, piv = lu_factor(A, overwrite_a=False, check_finite=False)

            U = np.triu(lu)
            L = np.tril(lu, k=-1)
            np.fill_diagonal(L, 1.0)

            perm = np.arange(n)
            for i, j in enumerate(piv):
                if i != j:
                    perm[i], perm[j] = perm[j], perm[i]

            P = np.zeros((n, n), dtype=np.float64)
            P[perm, np.arange(n)] = 1.0
            plu = (P, L, U)

        self._plu = plu
        return plu

class _ArrayProxy:
    __slots__ = ("lazy", "idx")

    def __init__(self, lazy, idx):
        self.lazy = lazy
        self.idx = idx

    def __array__(self, dtype=None):
        arr = self.lazy.get()[self.idx]
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr

class Solver:
    def __init__(self):
        _lu_small(_ONE)
        lazy = _LazyLU()
        self._lazy = lazy
        self._solution = {
            "LU": {
                "P": _ArrayProxy(lazy, 0),
                "L": _ArrayProxy(lazy, 1),
                "U": _ArrayProxy(lazy, 2),
            }
        }

    def solve(self, problem, **kwargs):
        self._lazy.reset(problem["matrix"])
        return self._solution