from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

def _topk_from_dense(A: NDArray, k: int) -> List[NDArray]:
    vals, vecs = np.linalg.eig(A)
    order = np.argsort(-np.abs(vals), kind="mergesort")
    if k < order.shape[0]:
        order = order[:k]
    return [vecs[:, i] for i in order]

def _sorted_vecs_by_abs(vals: NDArray, vecs: NDArray) -> List[NDArray]:
    order = np.argsort(-np.abs(vals), kind="mergesort")
    return [vecs[:, i] for i in order]

class Solver:
    __slots__ = ("_dense_n_max", "_v0_cache")

    def __init__(self) -> None:
        self._dense_n_max = 96
        # Cache deterministic start vectors; avoids allocating O(n) ones repeatedly.
        self._v0_cache: Dict[Tuple[int, str], NDArray] = {}

    @staticmethod
    def _is_exact_symmetric_real(A: sparse.spmatrix) -> bool:
        # Only for real dtypes. Exact structural/value equality check.
        diff = A - A.T
        if diff.nnz == 0:
            return True
        diff.eliminate_zeros()
        return diff.nnz == 0

    def _v0(self, n: int, dtype: np.dtype) -> NDArray:
        key = (n, dtype.str)
        v = self._v0_cache.get(key)
        if v is None:
            v = np.ones(n, dtype=dtype)
            self._v0_cache[key] = v
        return v

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        A = problem["matrix"]
        k = int(problem.get("k", 5))
        if k <= 0:
            return []

        # Problem instances are expected to give CSR already; keep conversions minimal.
        if sparse.isspmatrix(A):
            if not isinstance(A, sparse.csr_matrix):
                A = A.tocsr(copy=False)
            n = int(A.shape[0])
            if n == 0:
                return []
            if k >= n - 1:
                return _topk_from_dense(A.toarray(), min(k, n))
            if n <= self._dense_n_max:
                return _topk_from_dense(A.toarray(), k)
        else:
            A = np.asarray(A)
            n = int(A.shape[0])
            if n == 0:
                return []
            if n <= self._dense_n_max or k >= n - 1:
                return _topk_from_dense(A, min(k, n))
            A = sparse.csr_matrix(A)
            n = int(A.shape[0])

        # Deterministic start vector (match reference).
        v0 = self._v0(n, A.dtype)

        # k is fixed to 5 in this benchmark, so ncv is almost always 20.
        ncv = 20 if k <= 9 else max(2 * k + 1, 20)
        if ncv >= n:
            ncv = n - 1

        # Only attempt eigsh on exact real-symmetric matrices (cheaper check + avoids
        # complex Hermitian check overhead).
        if A.dtype.kind != "c" and self._is_exact_symmetric_real(A):
            vals, vecs = sparse.linalg.eigsh(
                A,
                k=k,
                v0=v0,
                maxiter=n * 200,
                ncv=ncv,
                which="LM",
            )
            return _sorted_vecs_by_abs(vals, vecs)

        vals, vecs = sparse.linalg.eigs(
            A,
            k=k,
            v0=v0,
            maxiter=n * 200,
            ncv=ncv,
        )
        return _sorted_vecs_by_abs(vals, vecs)