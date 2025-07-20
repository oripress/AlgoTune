import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

class Solver:
    def solve(self, problem, **kwargs):
        # Validate input
        try:
            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            shape = problem["shape"]
            sources = problem["source_indices"]
        except (KeyError, TypeError):
            return {"distances": []}
        if not isinstance(sources, (list, tuple)) or not sources:
            return {"distances": []}

        # Build CSR graph
        graph = csr_matrix((data, indices, indptr), shape=tuple(shape))

        # Multi-source Dijkstra (collapsed distances)
        dist = dijkstra(csgraph=graph, directed=False,
                        indices=sources, min_only=True)

        # Convert to Python list of floats (inf for unreachable)
        return {"distances": [dist.tolist()]}