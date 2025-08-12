from typing import Any

import numpy as np

try:
    from scipy.linalg import eigh as scipy_eigh
    from scipy.sparse.linalg import eigsh as sparse_eigsh, ArpackNoConvergence

    _HAVE_SCIPY = True
    _HAVE_SPARSE_EIGSH = True
except Exception:
    scipy_eigh = None
    sparse_eigsh = None
    ArpackNoConvergence = Exception  # placeholder
    _HAVE_SCIPY = False
    _HAVE_SPARSE_EIGSH = False

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the sparse PCA problem:
            minimize    ||B - X||_F^2 + λ ||X||_1
            subject to  ||X_i||_2 ≤ 1  for i=1,...,k

        Uses a closed-form per-column solution with soft-thresholding and
        unit-norm constraint handling per column.
        """
        try:
            # Use C-contiguous for small matrices to avoid copy; Fortran for large (faster LAPACK)
            SMALL_N = 96
            A_in = problem["covariance"]
            n_components = int(problem["n_components"])
            sparsity_param = float(problem["sparsity_param"])

            A0 = np.asarray(A_in, dtype=np.float64)
            n = A0.shape[0]
            if A0.shape[0] != A0.shape[1] or n_components <= 0:
                return {"components": [], "explained_variance": []}

            if n > SMALL_N:
                A = np.array(A0, order="F", copy=True)
            else:
                A = A0  # avoid extra copy for small problems

            # Compute top eigenpairs efficiently
            k_req = min(n_components, n)
            use_sparse = (
                _HAVE_SPARSE_EIGSH
                and 0 < k_req < n - 1
                and n >= 192
                and k_req <= min(16, max(1, n // 20))
            )
            if use_sparse:
                try:
                    # ARPACK Lanczos: good for large n and small k
                    ncv = min(n, max(2 * k_req + 2, 20))
                    w, v = sparse_eigsh(A, k=k_req, which="LM", tol=1e-6, ncv=ncv)
                except ArpackNoConvergence:
                    # Fallback to dense subset
                    if _HAVE_SCIPY:
                        w, v = scipy_eigh(
                            A,
                            subset_by_index=(n - k_req, n - 1),
                            overwrite_a=True,
                            check_finite=False,
                            driver="evr",
                        )
                    else:
                        w, v = np.linalg.eigh(A)
            elif _HAVE_SCIPY and 0 < k_req < n:
                # Largest k_req eigenpairs via LAPACK dsyevr
                w, v = scipy_eigh(
                    A,
                    subset_by_index=(n - k_req, n - 1),
                    overwrite_a=True,
                    check_finite=False,
                    driver="evr",
                )
            else:
                w, v = np.linalg.eigh(A)

            # Sort descending (eigh returns ascending; eigsh may be unsorted)
            if w.size > 0:
                idx_desc = np.argsort(w)[::-1]
                w = w[idx_desc]
                v = v[:, idx_desc]
            else:
                w = np.array([])
                v = np.zeros((n, 0), dtype=np.float64)

            # Keep strictly positive eigenvalues
            if w.size:
                pos_mask = w > 0.0
                if not np.all(pos_mask):
                    w = w[pos_mask]
                    v = v[:, pos_mask]

            # Determine number of usable components
            k = min(w.size, n_components)

            # Prepare output arrays
            X = np.zeros((n, n_components), dtype=np.float64)
            explained_variance = [0.0] * n_components
            if k == 0:
                return {"components": X.tolist(), "explained_variance": explained_variance}

            # Build scaled eigenvectors for the first k columns
            sqrtw = np.sqrt(w[:k], dtype=np.float64)
            Xk = v[:, :k] * sqrtw  # temporary matrix holding B[:, :k]

            # Closed-form per column via soft-threshold, then enforce l2 <= 1
            tau = 0.5 * sparsity_param
            if tau > 0.0:
                # Soft-threshold: Xk = sign(Xk) * max(|Xk| - tau, 0)
                absXk = np.abs(Xk)
                absXk -= tau
                np.maximum(absXk, 0.0, out=absXk)
                signXk = np.sign(Xk)
                Xk = absXk * signXk  # reuse arrays

            # Columnwise scaling for unit norm constraint (only when needed)
            norm2_k = np.sum(Xk * Xk, axis=0, dtype=np.float64)
            over_k = norm2_k > 1.0
            if np.any(over_k):
                Xk[:, over_k] /= np.sqrt(norm2_k[over_k])

            # Place into full X
            X[:, :k] = Xk

            # If all columns zero
            if not np.any(Xk):
                return {
                    "components": X.tolist(),
                    "explained_variance": explained_variance,
                }

            # Compute explained variance only for nonzero columns (among first k)
            nz_local = np.flatnonzero(norm2_k > 0.0)
            m = nz_local.size
            if m == 1:
                j = int(nz_local[0])
                xj = Xk[:, j]
                Ax = A @ xj
                explained_variance[j] = float(np.dot(xj, Ax))
            elif m == 2:
                j0 = int(nz_local[0])
                j1 = int(nz_local[1])
                x0 = Xk[:, j0]
                x1 = Xk[:, j1]
                Ax0 = A @ x0
                Ax1 = A @ x1
                explained_variance[j0] = float(np.dot(x0, Ax0))
                explained_variance[j1] = float(np.dot(x1, Ax1))
            elif m > 0:
                AX = A @ Xk[:, nz_local]
                ev = np.einsum("ij,ij->j", Xk[:, nz_local], AX, optimize=True)
                for idx, val in zip(nz_local.tolist(), ev.tolist()):
                    explained_variance[idx] = float(val)

            return {"components": X.tolist(), "explained_variance": explained_variance}
        except Exception:
            return {"components": [], "explained_variance": []}