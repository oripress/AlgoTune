from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

class Solver:
    """
    Fast APSP for sparse weighted graphs in CSR.

    Optimizations:
    - Use C-accelerated SciPy csgraph routines.
    - Use ndarray.tolist() (C-level conversion).
    - Feed preferred dtypes to reduce internal casting.
    """

    __slots__ = ()

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        try:
            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            shape = problem["shape"]
            n = int(shape[0])
        except Exception:
            return {"distance_matrix": []}

        if n <= 0:
            return {"distance_matrix": []}

        asarray = np.asarray
        csr_matrix = sp.csr_matrix
        floyd_warshall = csgraph.floyd_warshall
        shortest_path = csgraph.shortest_path

        try:
            data_a = asarray(data, dtype=np.float64)
            indices_a = asarray(indices, dtype=np.int32)
            indptr_a = asarray(indptr, dtype=np.int32)
            graph_csr = csr_matrix((data_a, indices_a, indptr_a), shape=shape, copy=False)
        except Exception:
            return {"distance_matrix": []}

        try:
            if n <= 96:
                dist = floyd_warshall(csgraph=graph_csr, directed=False)
            else:
                d = graph_csr.data
                unweighted = False
                if d.size:
                    mn = d.min()
                    if mn == 1.0 and d.max() == 1.0:
                        unweighted = True

                dist = shortest_path(
                    csgraph=graph_csr,
                    method="D",
                    directed=False,
                    unweighted=unweighted,
                    return_predecessors=False,
                )
        except Exception:
            return {"distance_matrix": []}

        return {"distance_matrix": dist.tolist()}