import numpy as np
from scipy.linalg.lapack import dgesv as _dgesv
import math
import numba as nb

@nb.njit(cache=True)
def _expm_2x2(a, b, c, d):
    s = (a + d) * 0.5
    p = (a - d) * 0.5
    delta_sq = p * p + b * c
    exp_s = math.exp(s)
    result = np.empty((2, 2))
    if abs(delta_sq) < 1e-30:
        result[0, 0] = exp_s * (1.0 + p)
        result[0, 1] = exp_s * b
        result[1, 0] = exp_s * c
        result[1, 1] = exp_s * (1.0 - p)
    elif delta_sq > 0:
        delta = math.sqrt(delta_sq)
        ch = math.cosh(delta)
        shd = math.sinh(delta) / delta
        result[0, 0] = exp_s * (ch + p * shd)
        result[0, 1] = exp_s * b * shd
        result[1, 0] = exp_s * c * shd
        result[1, 1] = exp_s * (ch - p * shd)
    else:
        delta = math.sqrt(-delta_sq)
        co = math.cos(delta)
        sid = math.sin(delta) / delta
        result[0, 0] = exp_s * (co + p * sid)
        result[0, 1] = exp_s * b * sid
        result[1, 0] = exp_s * c * sid
        result[1, 1] = exp_s * (co - p * sid)
    return result

@nb.njit(cache=True)
def _onenorm(A):
    n = A.shape[0]
    result = 0.0
    for j in range(n):
        col_sum = 0.0
        for i in range(n):
            col_sum += abs(A[i, j])
        if col_sum > result:
            result = col_sum
    return result

@nb.njit(cache=True)
def _add_diag(A, val):
    n = A.shape[0]
    for i in range(n):
        A[i, i] += val

@nb.njit(cache=True)
def _gauss_solve(A, B):
    n = A.shape[0]
    for k in range(n):
        max_val = abs(A[k, k])
        max_row = k
        for i in range(k + 1, n):
            if abs(A[i, k]) > max_val:
                max_val = abs(A[i, k])
                max_row = i
        if max_row != k:
            for j in range(k, n):
                A[k, j], A[max_row, j] = A[max_row, j], A[k, j]
            for j in range(n):
                B[k, j], B[max_row, j] = B[max_row, j], B[k, j]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            for j in range(k + 1, n):
                A[i, j] -= factor * A[k, j]
            A[i, k] = 0.0
            for j in range(n):
                B[i, j] -= factor * B[k, j]
    for k in range(n - 1, -1, -1):
        for j in range(n):
            for i in range(k + 1, n):
                B[k, j] -= A[k, i] * B[i, j]
            B[k, j] /= A[k, k]
    return B

@nb.njit(cache=True)
def _expm_numba(A):
    n = A.shape[0]
    norm_A = _onenorm(A)
    if norm_A == 0.0:
        return np.eye(n)
    I = np.eye(n)
    if norm_A <= 0.015:
        A2 = A @ A
        U = A @ (A2 + 60.0 * I)
        V = 12.0 * A2
        _add_diag(V, 120.0)
        return _gauss_solve(V - U, V + U)
    if norm_A <= 0.25:
        A2 = A @ A
        A4 = A2 @ A2
        U = A @ (A4 + 420.0 * A2 + 15120.0 * I)
        V = 30.0 * A4 + 3360.0 * A2
        _add_diag(V, 30240.0)
        return _gauss_solve(V - U, V + U)
    if norm_A <= 0.95:
        A2 = A @ A
        A4 = A2 @ A2
        A6 = A2 @ A4
        U = A @ (A6 + 1512.0 * A4 + 277200.0 * A2 + 8648640.0 * I)
        V = 56.0 * A6 + 25200.0 * A4 + 1995840.0 * A2
        _add_diag(V, 17297280.0)
        return _gauss_solve(V - U, V + U)
    if norm_A <= 2.1:
        A2 = A @ A
        A4 = A2 @ A2
        A6 = A2 @ A4
        A8 = A4 @ A4
        U = A @ (A8 + 3960.0 * A6 + 2162160.0 * A4 + 302702400.0 * A2 + 8821612800.0 * I)
        V = 90.0 * A8 + 110880.0 * A6 + 30270240.0 * A4 + 2075673600.0 * A2
        _add_diag(V, 17643225600.0)
        return _gauss_solve(V - U, V + U)
    s = max(0, int(math.ceil(math.log(norm_A / 5.4) * 1.4426950408889634)))
    if s > 0:
        A = A / (2.0 ** s)
    b0 = 64764752532480000.
    b1 = 32382376266240000.
    b2 = 7771770303897600.
    b3 = 1187353796428800.
    b4 = 129060195264000.
    b5 = 10559470521600.
    b6 = 670442572800.
    b7 = 33522128640.
    b8 = 1323241920.
    b9 = 40840800.
    b10 = 960960.
    b11 = 16380.
    b12 = 182.
    b13 = 1.
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A2 @ A4
    W1 = b13 * A6 + b11 * A4 + b9 * A2
    W2 = A6 @ W1 + b7 * A6 + b5 * A4 + b3 * A2
    _add_diag(W2, b1)
    U = A @ W2
    Z1 = b12 * A6 + b10 * A4 + b8 * A2
    V = A6 @ Z1 + b6 * A6 + b4 * A4 + b2 * A2
    _add_diag(V, b0)
    R = _gauss_solve(V - U, V + U)
    for _ in range(s):
        R = R @ R
    return R

def _solve_pade_lapack(VmU, VpU):
    if not VmU.flags['F_CONTIGUOUS']:
        VmU = np.asfortranarray(VmU)
    if not VpU.flags['F_CONTIGUOUS']:
        VpU = np.asfortranarray(VpU)
    _, _, X, _ = _dgesv(VmU, VpU, overwrite_a=1, overwrite_b=1)
    return X

def _expm_pade_large(A, n):
    norm_A = _onenorm(A)
    if norm_A == 0:
        return np.eye(n, dtype=np.float64)
    di = np.diag_indices(n)
    s = 0
    if norm_A > 5.4:
        s = int(math.ceil(math.log(norm_A / 5.4) * 1.4426950408889634))
        A = A * (2.0 ** (-s))
    elif norm_A <= 2.1:
        A2 = A @ A
        if norm_A <= 0.015:
            ident = np.eye(n, dtype=np.float64)
            U = A @ (A2 + 60.0 * ident)
            V = 12.0 * A2
            V[di] += 120.0
            return _solve_pade_lapack(V - U, V + U)
        A4 = A2 @ A2
        if norm_A <= 0.25:
            tmp = A4 + 420.0 * A2
            tmp[di] += 15120.0
            U = A @ tmp
            V = 30.0 * A4 + 3360.0 * A2
            V[di] += 30240.0
            return _solve_pade_lapack(V - U, V + U)
        A6 = A2 @ A4
        if norm_A <= 0.95:
            tmp = A6 + 1512.0 * A4 + 277200.0 * A2
            tmp[di] += 8648640.0
            U = A @ tmp
            V = 56.0 * A6 + 25200.0 * A4 + 1995840.0 * A2
            V[di] += 17297280.0
            return _solve_pade_lapack(V - U, V + U)
        A8 = A4 @ A4
        tmp = A8 + 3960.0 * A6 + 2162160.0 * A4 + 302702400.0 * A2
        tmp[di] += 8821612800.0
        U = A @ tmp
        V = 90.0 * A8 + 110880.0 * A6 + 30270240.0 * A4 + 2075673600.0 * A2
        V[di] += 17643225600.0
        return _solve_pade_lapack(V - U, V + U)
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A2 @ A4
    W1 = A6 + 16380.0 * A4
    W1 += 40840800.0 * A2
    W2 = A6 @ W1
    W2 += 33522128640.0 * A6
    W2 += 10559470521600.0 * A4
    W2 += 1187353796428800.0 * A2
    W2[di] += 32382376266240000.0
    U = A @ W2
    Z1 = 182.0 * A6 + 960960.0 * A4
    Z1 += 1323241920.0 * A2
    V = A6 @ Z1
    V += 670442572800.0 * A6
    V += 129060195264000.0 * A4
    V += 7771770303897600.0 * A2
    V[di] += 64764752532480000.0
    R = _solve_pade_lapack(V - U, V + U)
    for _ in range(s):
        R = R @ R
    return R

# Pre-compile numba functions
_dummy1 = _expm_2x2(1.0, 0.0, 0.0, 1.0)
_dummy2 = _onenorm(np.eye(2))
_dummy3 = _expm_numba(np.eye(3))

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["matrix"]
        if not isinstance(A, np.ndarray):
            A = np.array(A, dtype=np.float64)
        elif A.dtype != np.float64:
            A = A.astype(np.float64)
        n = A.shape[0]
        if n == 1:
            return {"exponential": np.array([[math.exp(A[0, 0])]])}
        if n == 2:
            return {"exponential": _expm_2x2(A[0, 0], A[0, 1], A[1, 0], A[1, 1])}
        if n <= 16:
            return {"exponential": _expm_numba(A)}
        return {"exponential": _expm_pade_large(A, n)}