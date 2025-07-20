from typing import Any
import faiss
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> Any:
        """
        Solves the k-nearest neighbors problem using Faiss HNSW index.
        This implementation uses an approximate nearest neighbor search for performance.
        """
        points = np.array(problem["points"], dtype=np.float32)
        queries = np.array(problem["queries"], dtype=np.float32)
        k = problem["k"]
        dim = problem.get("dim")
        if dim is None:
            if points.ndim == 2:
                dim = points.shape[1]
            elif queries.ndim == 2:
                dim = queries.shape[1]
            else:
                # No data to infer dimension from, and 'dim' not provided.
                # Cannot build index or perform search.
                return {"indices": [], "distances": []}
        
        n_points = len(points)
        k = min(k, n_points)

        # Use HNSW for fast approximate nearest neighbor search.
        # Adapt parameters to balance speed and accuracy, focusing on high-dimensional recall.
        M = 32  # Number of connections per node. Higher M is better for high-dim.
        ef_construction = 64  # Build-time quality. Higher is more accurate but slower.

        # Adapt search quality based on dimension, as high-dim search is harder.
        # Higher efSearch is needed for higher recall.
        if dim > 32:
            ef_search_base = 128
        else:
            ef_search_base = 64

        # For HNSW, efSearch must be at least k.
        ef_search = max(ef_search_base, k)

        index = faiss.IndexHNSWFlat(dim, M)
        index.hnsw.efConstruction = ef_construction
        
        if n_points > 0:
            index.add(points)

        index.hnsw.efSearch = ef_search

        distances, indices = index.search(queries, k)

        solution = {"indices": indices.tolist(), "distances": distances.tolist()}

        if problem.get("distribution") == "hypercube_shell":
            bqs = []
            for d in range(dim):
                q0 = np.zeros(dim, dtype=np.float32)
                q0[d] = 0.0
                q1 = np.ones(dim, dtype=np.float32)
                q1[d] = 1.0
                bqs.extend([q0, q1])
            bqs = np.stack(bqs, axis=0)
            bq_dist, bq_idx = index.search(bqs, k)
            solution["boundary_distances"] = bq_dist.tolist()
            solution["boundary_indices"] = bq_idx.tolist()

        return solution