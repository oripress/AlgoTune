import numpy as np
import scipy.linalg as la
from numba import njit

try:
    from scipy.linalg._matfuncs_sqrtm import _sqrtm_triu
except Exception:
    _sqrtm_triu = None

_EMPTY_SOLUTION = {"sqrtm": {"X": []}}
_RTOL = 1e-5
_ATOL = 1e-8

@njit(cache=True)
def _sqrt_upper_triangular(T):
    n = T.shape[0]
    U = np.zeros((n, n), dtype=np.complex128)

    for i in range(n):
        U[i, i] = np.sqrt(T[i, i])

    for gap in range(1, n):
        for i in range(n - gap):
            j = i + gap
            s = T[i, j]
            for k in range(i + 1, j):
                s -= U[i, k] * U[k, j]
            denom = U[i, i] + U[j, j]
            scale = abs(U[i, i]) + abs(U[j, j]) + 1.0
            if abs(denom) < 1e-12 * scale:
                return U, False
            U[i, j] = s / denom

    return U, True

@njit(cache=True)
def _classify_matrix(A):
    n = A.shape[0]
    has_lower = False
    has_upper = False
    hermitian = True
    finite = True
    any_nonzero = False

    for i in range(n):
        for j in range(n):
            z = A[i, j]
            if z.real != 0.0 or z.imag != 0.0:
                any_nonzero = True
            if finite and (not np.isfinite(z.real) or not np.isfinite(z.imag)):
                finite = False
            if i > j:
                if z.real != 0.0 or z.imag != 0.0:
                    has_lower = True
            elif i < j:
                if z.real != 0.0 or z.imag != 0.0:
                    has_upper = True
            if hermitian:
                w = A[j, i]
                if z.real != w.real or z.imag != -w.imag:
                    hermitian = False

    return has_lower, has_upper, hermitian, finite, any_nonzero

def _wrap(X):
    return {"sqrtm": {"X": X.tolist()}}

def _valid_square_root(A, X):
    return np.isfinite(X).all() and np.allclose(X @ X, A, rtol=_RTOL, atol=_ATOL)

class Solver:
    def __init__(self):
        dummy = np.zeros((1, 1), dtype=np.complex128)
        _sqrt_upper_triangular(dummy)
        _classify_matrix(dummy)

    def solve(self, problem, **kwargs):
        try:
            A = np.asarray(problem["matrix"], dtype=np.complex128)

            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                return _EMPTY_SOLUTION

            n = A.shape[0]

            if n == 0:
                return _EMPTY_SOLUTION

            if n == 1:
                z = A[0, 0]
                if not (np.isfinite(z.real) and np.isfinite(z.imag)):
                    return _EMPTY_SOLUTION
                return _wrap(np.array([[np.sqrt(z)]], dtype=np.complex128))

            has_lower, has_upper, hermitian, finite, any_nonzero = _classify_matrix(A)

            if not finite:
                return _EMPTY_SOLUTION

            if not any_nonzero:
                return _wrap(np.zeros_like(A))

            if n == 2:
                delta = np.sqrt(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
                tau = np.sqrt(A[0, 0] + A[1, 1] + 2.0 * delta)
                if abs(tau) > 1e-14:
                    X = (A + delta * np.eye(2, dtype=np.complex128)) / tau
                    if _valid_square_root(A, X):
                        return _wrap(X)

            if not has_lower:
                if not has_upper:
                    return _wrap(np.diag(np.sqrt(np.diag(A))))

                if _sqrtm_triu is not None:
                    U = np.asarray(_sqrtm_triu(A), dtype=np.complex128)
                    if np.isfinite(U).all():
                        return _wrap(U)

                U, ok = _sqrt_upper_triangular(A)
                if ok and np.isfinite(U).all():
                    return _wrap(U)

            if not has_upper:
                At = A.T.copy()

                if _sqrtm_triu is not None:
                    U = np.asarray(_sqrtm_triu(At), dtype=np.complex128)
                    L = U.T
                    if np.isfinite(L).all():
                        return _wrap(L)

                Ut, ok = _sqrt_upper_triangular(At)
                if ok:
                    L = Ut.T
                    if np.isfinite(L).all():
                        return _wrap(L)

            if hermitian:
                w, Q = np.linalg.eigh(A)
                X = (Q * np.sqrt(w.astype(np.complex128))) @ Q.conj().T
                if np.isfinite(X).all():
                    return _wrap(X)

            if n <= 6:
                try:
                    w, V = np.linalg.eig(A)
                    X = (V * np.sqrt(w)) @ np.linalg.inv(V)
                    if _valid_square_root(A, X):
                        return _wrap(X)
                except Exception:
                    pass

            T, Z = la.schur(A, output="complex", overwrite_a=False, check_finite=False)
            if _sqrtm_triu is not None:
                U = np.asarray(_sqrtm_triu(T), dtype=np.complex128)
                X = Z @ U @ Z.conj().T
                if np.isfinite(X).all():
                    return _wrap(X)
            else:
                U, ok = _sqrt_upper_triangular(T)
                if ok:
                    X = Z @ U @ Z.conj().T
                    if np.isfinite(X).all():
                        return _wrap(X)

            out = la.sqrtm(A, disp=False)
            X = out[0] if isinstance(out, tuple) else out
            X = np.asarray(X, dtype=np.complex128)
            if np.isfinite(X).all():
                return _wrap(X)

            return _EMPTY_SOLUTION
        except Exception:
            return _EMPTY_SOLUTION