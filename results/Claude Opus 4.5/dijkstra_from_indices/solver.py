import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

class Solver:
    def solve(self, problem, **kwargs):
        graph = csr_matrix(
            (problem["data"], problem["indices"], problem["indptr"]),
            shape=problem["shape"],
        )
        source_indices = problem["source_indices"]
        
        if not source_indices:
            return {"distances": []}
        
        dist = dijkstra(graph, directed=False, indices=source_indices, min_only=False)
        
        if dist.ndim == 1:
            result = [[(None if np.isinf(d) else d) for d in dist]]
        else:
            result = [[(None if np.isinf(d) else d) for d in row] for row in dist]
        
        return {"distances": result}