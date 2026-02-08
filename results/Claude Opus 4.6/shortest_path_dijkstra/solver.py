import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        try:
            data = np.array(problem["data"], dtype=np.float64)
            indices = np.array(problem["indices"], dtype=np.int32)
            indptr = np.array(problem["indptr"], dtype=np.int32)
            shape = tuple(problem["shape"])
            n = shape[0]
            
            graph_csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
            
            dist_matrix = scipy.sparse.csgraph.shortest_path(
                csgraph=graph_csr, method='D', directed=False
            )
            
            result = dist_matrix.tolist()
            for i in range(n):
                for j in range(n):
                    if result[i][j] == float('inf'):
                        result[i][j] = None
            
            return {"distance_matrix": result}
        except Exception:
            return {"distance_matrix": []}