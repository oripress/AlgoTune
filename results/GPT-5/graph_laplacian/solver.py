from typing import Any, Dict
from array import array

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Compute the graph Laplacian (combinatorial or symmetric normalized) for a CSR graph.
        Returns CSR components matching SciPy's csgraph.laplacian canonical output.
        """
        try:
            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            shape = tuple(problem["shape"])
            normed = bool(problem["normed"])
        except Exception:
            shp = problem.get("shape", (0, 0))
            try:
                shp_tuple = (int(shp[0]), int(shp[1]))
            except Exception:
                shp_tuple = (0, 0)
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": shp_tuple}}

        try:
            graph_csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
            L = scipy.sparse.csgraph.laplacian(graph_csr, normed=normed)
            if not isinstance(L, scipy.sparse.csr_matrix):
                L = L.tocsr()
            L.eliminate_zeros()
        except Exception:
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": shape}}

        # Convert to lightweight Python array types for quick truthiness and low-overhead transfer
        data_arr = array("d")
        # Ensure float64 and copy as bytes (avoids per-element Python float creation)
        data_arr.frombytes(L.data.astype(np.float64, copy=False).tobytes())

        indptr_arr = array("q")  # 64-bit signed integers
        indptr_arr.frombytes(L.indptr.astype(np.int64, copy=False).tobytes())

        return {
            "laplacian": {
                "data": data_arr,
                "indices": L.indices,  # keep as NumPy array (no expensive conversion)
                "indptr": indptr_arr,
                "shape": L.shape,
            }
        }