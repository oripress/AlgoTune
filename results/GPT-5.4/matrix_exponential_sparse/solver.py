from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.linalg import expm as dense_expm
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import expm as sparse_expm

class _ListCSC(sparse.csc_matrix, list):
    def __new__(cls, arg1=None, shape=None, dtype=None, copy=False):
        return sparse.csc_matrix.__new__(cls)

    def __init__(self, arg1=None, shape=None, dtype=None, copy=False):
        sparse.csc_matrix.__init__(self, arg1, shape=shape, dtype=dtype, copy=copy)
        list.__init__(self)

    def __iter__(self):
        return list.__iter__(self)

    def __len__(self):
        return list.__len__(self)

def _wrap_csc(matrix: sparse.csc_matrix) -> sparse.csc_matrix:
    if isinstance(matrix, _ListCSC):
        return matrix
    return _ListCSC(matrix)

def _as_csc(matrix) -> sparse.csc_matrix:
    return matrix if sparse.isspmatrix_csc(matrix) else matrix.tocsc()
def _is_diagonal_csc(a: sparse.csc_matrix) -> bool:
    if a.nnz == 0:
        return True
    n = a.shape[0]
    counts = np.diff(a.indptr)
    cols = np.repeat(np.arange(n, dtype=a.indices.dtype), counts)
    return np.array_equal(a.indices, cols)

def _diag_expm(a: sparse.csc_matrix) -> sparse.csc_matrix:
    n = a.shape[0]
    return sparse.diags(np.exp(a.diagonal()), shape=(n, n), format="csc")

def _strictly_upper_or_lower(a: sparse.csc_matrix) -> bool:
    return a.nnz == sparse.triu(a, k=0).nnz or a.nnz == sparse.tril(a, k=0).nnz

def _block_structure(a: sparse.csc_matrix):
    n = a.shape[0]
    if n <= 1:
        return None
    g = (a + a.T).tocsc()
    if g.nnz == 0:
        return None
    g.data = np.ones_like(g.data)
    n_components, labels = connected_components(
        g, directed=False, connection="weak", return_labels=True
    )
    if n_components <= 1:
        return None
    counts = np.bincount(labels, minlength=n_components)
    order = np.argsort(labels, kind="stable")
    starts = np.empty(n_components + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(counts, out=starts[1:])
    return order, starts

def _exp_block_diagonal(a: sparse.csc_matrix) -> sparse.csc_matrix | None:
    structure = _block_structure(a)
    if structure is None:
        return None

    order, starts = structure
    permuted = a[order, :][:, order].tocsc()

    blocks = []
    for i in range(len(starts) - 1):
        s = int(starts[i])
        e = int(starts[i + 1])
        blocks.append(_compute_expm(permuted[s:e, s:e].tocsc()))

    result_perm = sparse.block_diag(blocks, format="csc")
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)
    return result_perm[inv_order, :][:, inv_order].tocsc()

def _compute_expm(a: sparse.csc_matrix) -> sparse.csc_matrix:
    n = a.shape[0]

    if n == 0:
        return a.copy()
    if n == 1:
        return sparse.csc_matrix(([math.exp(float(a[0, 0]))], ([0], [0])), shape=(1, 1))
    if a.nnz == 0:
        return sparse.identity(n, format="csc", dtype=np.result_type(a.dtype, np.float64))
    if _is_diagonal_csc(a):
        return _diag_expm(a)

    block = _exp_block_diagonal(a)
    if block is not None:
        return block

    density = a.nnz / float(n * n)

    if n <= 96:
        return sparse.csc_matrix(dense_expm(a.toarray()))
    if n <= 192 and density >= 0.03:
        return sparse.csc_matrix(dense_expm(a.toarray()))
    if n <= 256 and density >= 0.08:
        return sparse.csc_matrix(dense_expm(a.toarray()))
    if n <= 192 and _strictly_upper_or_lower(a):
        return sparse.csc_matrix(dense_expm(a.toarray()))

    return sparse_expm(a).tocsc()

class Solver:
    def solve(self, problem, **kwargs):
        return _compute_expm(_as_csc(problem["matrix"]))