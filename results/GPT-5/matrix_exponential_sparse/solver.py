from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm
from scipy.linalg import expm as _dense_expm
class _LenFriendlyCSC(sparse.csc_matrix):
    # Some harnesses call len(...) on the returned sparse matrix; default raises TypeError.
    def __len__(self) -> int:  # type: ignore[override]
        return self.shape[0]

class Solver:
    def solve(self, problem: dict[str, sparse.spmatrix], **kwargs: Any) -> sparse.spmatrix:
        """
        Compute the matrix exponential of a square sparse matrix A (CSC format).

        Fast paths:
        - Zero matrix -> Identity
        - Diagonal matrix -> exp of diagonal

        Fallback:
        - scipy.sparse.linalg.expm for general sparse matrices.

        Returns a CSC sparse matrix (with a len(...) that returns nrows).
        """
        A_in = problem["matrix"]

        # Force CSC MATRIX (not sparse array) to avoid ambiguous behaviors
        A = sparse.csc_matrix(A_in, copy=False)

        # Ensure sorted indices for predictable behavior/performance
        A.sort_indices()

        n, m = A.shape
        if n != m:
            raise ValueError("Input matrix must be square.")

        # Zero matrix -> identity (ensure csc_matrix, not sparse array)
        if A.nnz == 0:
            diag_idx = np.arange(n, dtype=np.int64)
            data = np.ones(n, dtype=A.dtype)
            return _LenFriendlyCSC((data, (diag_idx, diag_idx)), shape=(n, n))

        # Prepare counts and per-column row range (min,max)
        indptr = A.indptr
        indices = A.indices
        counts = np.diff(indptr)

        row_min = np.empty(n, dtype=np.int64)
        row_max = np.empty(n, dtype=np.int64)
        for j in range(n):
            start, end = indptr[j], indptr[j + 1]
            if start == end:
                # empty column
                row_min[j] = j
                row_max[j] = j
            else:
                # indices are sorted after A.sort_indices()
                row_min[j] = int(indices[start])
                row_max[j] = int(indices[end - 1])

        # Fast path: diagonal-only (all nonzeros on the diagonal)
        arange_n = np.arange(n, dtype=np.int64)
        if np.all((counts == 0) | ((row_min == arange_n) & (row_max == arange_n))):
            # Aggregate any duplicates per column (diagonal position)
            diag_vals = np.zeros(n, dtype=A.dtype)
            data = A.data
            # Efficient per-column sum using reduceat
            if data.size:
                col_sums = np.add.reduceat(data, indptr[:-1])
                # For columns with zero nnz, reduceat doesn't write; fill only where counts>0
                has_nnz = counts > 0
                diag_vals[has_nnz] = col_sums[has_nnz]
            # Exponentiate diagonal and build explicit csc_matrix
            diag_exp = np.exp(diag_vals)
            idx = np.arange(n, dtype=np.int64)
            return _LenFriendlyCSC((diag_exp, (idx, idx)), shape=(n, n))

        # Fast path: detect contiguous block-diagonal structure using row_min/row_max
        blocks: list[tuple[int, int]] = []
        ok = True
        s = 0
        while s < n:
            end_b = max(s, row_max[s])
            j = s
            while j <= end_b:
                if row_min[j] < s:
                    ok = False
                    break
                if row_max[j] > end_b:
                    end_b = row_max[j]
                j += 1
            if not ok:
                break
            blocks.append((s, end_b))
            s = end_b + 1

        if ok and len(blocks) > 1:
            # Compute expm on each block independently and assemble block-diagonal result
            block_mats = []
            for s0, e0 in blocks:
                sub = A[s0 : e0 + 1, s0 : e0 + 1]
                if not sparse.isspmatrix_csc(sub):
                    sub = sub.tocsc(copy=False)
                Eb = expm(sub)
                if not sparse.isspmatrix_csc(Eb):
                    Eb = Eb.tocsc(copy=False)
                block_mats.append(Eb)
            E = sparse.block_diag(block_mats, format="csc")
            return _LenFriendlyCSC(E)

        # General case: use scipy's sparse expm
        E = expm(A)
        # Ensure CSC MATRIX
        if not sparse.isspmatrix_csc(E):
            E = E.tocsc(copy=False)
        return _LenFriendlyCSC(E)