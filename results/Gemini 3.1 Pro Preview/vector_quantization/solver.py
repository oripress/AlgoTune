import faiss
import numpy as np
from typing import Any

faiss.omp_set_num_threads(1)

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        vectors = np.array(problem["vectors"], dtype=np.float32)
        k = problem["k"]
        dim = vectors.shape[1]

        kmeans = faiss.Kmeans(dim, k, niter=58, verbose=False)
        kmeans.train(vectors)
        centroids = kmeans.centroids

        distances, assignments = kmeans.index.search(vectors, 1)

        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.ravel().tolist(),
            "quantization_error": distances.mean().item(),
        }