import numpy as np
import scipy.sparse
import scipy.sparse.csgraph as csgraph

class Solver:
    def solve(self, problem, **kwargs):
        # Convert CSR components to NumPy arrays
        data = np.array(problem["data"], dtype=float)
        indices = np.array(problem["indices"], dtype=np.int32)
        indptr = np.array(problem["indptr"], dtype=np.int32)
        n = problem["shape"][0]
        # Build CSR matrix
        graph = scipy.sparse.csr_matrix((data, indices, indptr), shape=(n, n))
        directed = kwargs.get("directed", False)
        method = kwargs.get("method", "D")
        # Use multi-source Dijkstra directly when method is 'D'
        if method == "D":
            # Single C call for all sources
            src = np.arange(n, dtype=np.int32)
            dist = csgraph.dijkstra(graph, directed=directed, indices=src, return_predecessors=False)
        else:
            # Fallback to shortest_path for other methods
            dist = csgraph.shortest_path(graph, directed=directed, method=method)
        # Convert to Python list of lists
        return {"distance_matrix": dist.tolist()}