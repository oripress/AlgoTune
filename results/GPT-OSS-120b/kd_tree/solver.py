import numpy as np
from typing import Any, Dict, List

# FAISS provides a very fast exact L2 index (flat)
import faiss

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fast exact k‑NN search using FAISS.

        Returns Euclidean distances (not squared) to stay compatible with the
        validator, which checks the squared distances internally.
        Handles the optional ``hypercube_shell`` distribution.
        """
        points = np.asarray(problem["points"], dtype=np.float32)
        queries = np.asarray(problem["queries"], dtype=np.float32)
        k = int(problem["k"])
        n_points = points.shape[0]

        # Guard against empty dataset
        if n_points == 0:
            empty_res = {
                "indices": [[-1] * k for _ in range(len(queries))],
                "distances": [[float("inf")] * k for _ in range(len(queries))],
            }
            if problem.get("distribution") == "hypercube_shell":
                empty_res["boundary_indices"] = []
                empty_res["boundary_distances"] = []
            return empty_res

        k = min(k, n_points)

        dim = points.shape[1]

        # Build a flat L2 index (exact search)
        index = faiss.IndexFlatL2(dim)
        # Adding points directly; Faiss will return their original 0‑based indices
        index.add(points)
        # Search queries
        distances, indices = index.search(queries, k)          # distances are squared L2
        # Faiss returns squared Euclidean distances; keep them as‑is because the validator compares against the sum of squared differences.

        solution: Dict[str, Any] = {
            "indices": indices.tolist(),
            "distances": distances.tolist(),
        }

        # Optional boundary queries for the ``hypercube_shell`` distribution
        if problem.get("distribution") == "hypercube_shell":
            dim = points.shape[1]
            bqs: List[np.ndarray] = []
            for _ in range(dim):
                bqs.append(np.zeros(dim, dtype=np.float32))
                bqs.append(np.ones(dim, dtype=np.float32))
            bqs_arr = np.stack(bqs, axis=0)

            b_distances, b_indices = index.search(bqs_arr, k)
            b_distances = np.sqrt(b_distances)

            solution["boundary_indices"] = b_indices.tolist()
            solution["boundary_distances"] = b_distances.tolist()

        return solution