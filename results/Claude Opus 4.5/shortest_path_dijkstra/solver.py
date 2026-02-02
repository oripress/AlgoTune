from typing import Any
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """Compute all-pairs shortest paths."""
        shape = tuple(problem["shape"])
        n = shape[0]
        
        if n == 0:
            return {"distance_matrix": []}
        
        data = problem["data"]
        indices = problem["indices"]
        indptr = problem["indptr"]
        num_edges = len(data)
        density = num_edges / (n * n) if n > 1 else 1
        
        # For small/dense graphs, build dense matrix directly
        if n < 64 or density > 0.1:
            # Build dense matrix directly - avoid CSR overhead
            dist = np.full((n, n), np.inf, dtype=np.float64)
            np.fill_diagonal(dist, 0.0)
            
            for i in range(n):
                for idx in range(indptr[i], indptr[i+1]):
                    j = indices[idx]
                    w = data[idx]
                    if w < dist[i, j]:
                        dist[i, j] = w
            
            # Make symmetric for undirected
            dist = np.minimum(dist, dist.T)
            
            dist_matrix = scipy.sparse.csgraph.floyd_warshall(
                dist, directed=False
            )
        else:
            # For larger sparse graphs, CSR is worth it
            data_arr = np.asarray(data, dtype=np.float64)
            indices_arr = np.asarray(indices, dtype=np.int32)
            indptr_arr = np.asarray(indptr, dtype=np.int32)
            graph_csr = scipy.sparse.csr_matrix((data_arr, indices_arr, indptr_arr), shape=shape)
            
            dist_matrix = scipy.sparse.csgraph.dijkstra(
                graph_csr, directed=False
            )
        
        # Fast inf replacement
        inf_mask = np.isinf(dist_matrix)
        if inf_mask.any():
            obj_arr = dist_matrix.astype(object)
            obj_arr[inf_mask] = None
            result = obj_arr.tolist()
        else:
            result = dist_matrix.tolist()
        
        return {"distance_matrix": result}