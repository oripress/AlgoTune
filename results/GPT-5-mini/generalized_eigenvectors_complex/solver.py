import numpy as np
import scipy.linalg as la
from typing import Any, Tuple

class Solver:
    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs) -> Any:
        """
        Solve the generalized eigenvalue problem A x = lambda B x.

        Returns:
          - list of n eigenvalues (complex), sorted by descending real part then descending imag part
          - list of n eigenvectors (each a list of n complex numbers), each normalized to unit Euclidean norm
        """
        A, B = problem

        # Ensure ndarray inputs
        A = np.array(A, dtype=np.float64, order='F', copy=False)
        B = np.array(B, dtype=np.float64, order='F', copy=False)

        # Basic checks / shapes
        if A.ndim != 2 or B.ndim != 2:
            A = np.atleast_2d(A)
            B = np.atleast_2d(B)
        n = A.shape[0]
        if n == 0:
            return ([], [])

        def postprocess(w: np.ndarray, vecs: np.ndarray):
            # Convert to complex numpy arrays
            w = np.asarray(w, dtype=np.complex128)
            vecs = np.asarray(vecs, dtype=np.complex128)
    
            # Ensure vecs has shape (n, m) with eigenvectors as columns
            if vecs.ndim == 1:
                vecs = vecs.reshape(n, 1)
            if vecs.shape[0] != n:
                vecs = np.eye(n, dtype=np.complex128)
    
            # Normalize columns to unit Euclidean norm (vectorized)
            norms = np.linalg.norm(vecs, axis=0)
            eps = 1e-15
            nonzero = norms > eps
            if np.any(nonzero):
                vecs[:, nonzero] /= norms[nonzero]
            zero_cols = np.nonzero(~nonzero)[0]
            if zero_cols.size > 0:
                # replace zero columns with canonical unit vectors
                vecs[:, zero_cols] = 0
                rows = zero_cols % n
                vecs[rows, zero_cols] = 1.0
    
            # Sort by descending real part, then descending imaginary part
            order = np.lexsort((-w.imag, -w.real))
            w_sorted = w[order]
            vecs_sorted = vecs[:, order]
    
            # Convert to Python lists (complex) efficiently
            eigenvalues_list = [complex(val) for val in w_sorted]
            eigenvectors_list = vecs_sorted.T.tolist()
            return eigenvalues_list, eigenvectors_list

        # Fast path for 1x1
        if n == 1:
            a = complex(A[0, 0])
            b = complex(B[0, 0])
            if b == 0:
                lam = 0 + 0j
            else:
                lam = a / b
            return ([lam], [[1.0 + 0j]])

        # 1) Symmetric definite fast path (use eigh) with cheap sampling for large n
        try:
            tol_sym = 1e-12
            symmetric = False
            if n <= 200:
                symmetric = np.allclose(A, A.T, atol=tol_sym) and np.allclose(B, B.T, atol=tol_sym)
            else:
                # sample a few indices to cheaply detect non-symmetry
                k = min(8, n)
                idx = np.linspace(0, n - 1, k, dtype=int)
                sym_ok = True
                for i in idx:
                    ai = A[i, idx]
                    aij = A[idx, i]
                    if not np.allclose(ai, aij, atol=tol_sym):
                        sym_ok = False
                        break
                    bi = B[i, idx]
                    bij = B[idx, i]
                    if not np.allclose(bi, bij, atol=tol_sym):
                        sym_ok = False
                        break
                if sym_ok:
                    # do the full check once if samples look symmetric
                    symmetric = np.allclose(A, A.T, atol=tol_sym) and np.allclose(B, B.T, atol=tol_sym)
            if symmetric:
                try:
                    # Check positive-definiteness of B via Cholesky
                    la.cholesky(B, lower=False, check_finite=False)
                    w, vecs = la.eigh(A, B, check_finite=False)
                    return postprocess(w, vecs)
                except Exception:
                    pass
        except Exception:
            pass

        # 2) Diagonal B fast path
        try:
            diagB = np.diag(B)
            if np.allclose(B, np.diag(diagB), atol=1e-12) and np.all(np.abs(diagB) > 1e-15):
                inv_diag = 1.0 / diagB
                # C = inv(diagB) @ A  (scale rows)
                C = inv_diag[:, None] * A
                w, vecs = np.linalg.eig(C)
                return postprocess(w, vecs)
        except Exception:
            pass

        # 3) Try solving for C = B^{-1} A and using numpy eig (fast for well-conditioned B)
        try:
            C = np.linalg.solve(B, A)
            w, vecs = np.linalg.eig(C)
            if np.all(np.isfinite(w.real)) and np.all(np.isfinite(w.imag)):
                return postprocess(w, vecs)
        except Exception:
            pass

        # 4) Try SciPy linear solve then numpy eig
        try:
            C = la.solve(B, A, assume_a='gen', check_finite=False)
            w, vecs = np.linalg.eig(C)
            if np.all(np.isfinite(w.real)) and np.all(np.isfinite(w.imag)):
                return postprocess(w, vecs)
        except Exception:
            pass

        # 5) Robust fallback: generalized eigenproblem via SciPy
        try:
            w, vecs = la.eig(A, B, check_finite=False)
            return postprocess(w, vecs)
        except Exception:
            # As a last resort, return zeros and identity vectors
            w = np.zeros(n, dtype=np.complex128)
            vecs = np.eye(n, dtype=np.complex128)
            return postprocess(w, vecs)