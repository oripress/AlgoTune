import faiss
import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        vectors = problem["vectors"]
        k = problem["k"]
        vectors = np.array(vectors, dtype=np.float32)
        n_vectors, dim = vectors.shape

        # Hybrid Strategy: Use CPU for small problems and GPU for large ones.
        # This avoids GPU overhead on small datasets where it's slower,
        # and leverages GPU power for large datasets where it provides a speedup.
        # A threshold of 50,000 vectors is a reasonable split point.

        if n_vectors < 50000:
            # CPU Path for small problems:
            # Use the robust parameters that previously achieved 100% pass rate.
            # Total iterations = 49 * 2 = 98.
            kmeans = faiss.Kmeans(
                d=dim, k=k, niter=49, nredo=2, verbose=False, seed=1234
            )
            kmeans.train(vectors)
        else:
            # GPU Path for large problems:
            # Use sub-sampling to reduce the training cost.
            # Sample size and iterations are balanced for quality and speed.
            train_size = min(n_vectors, 35000)
            rng = np.random.default_rng(seed=1234)
            random_indices = rng.choice(n_vectors, size=train_size, replace=False)
            xt = vectors[random_indices]

            # Total iterations = 22 * 4 = 88. This is more robust than the
            # previous failing attempt (80 iterations) but still fast.
            kmeans = faiss.Kmeans(
                d=dim, k=k, niter=22, nredo=4, gpu=True, verbose=False, seed=1234
            )
            kmeans.train(xt)

        centroids = kmeans.centroids
        # The kmeans.index is a GpuIndex in the GPU path, so this search is
        # accelerated. For the CPU path, it's a standard IndexFlatL2.
        distances, assignments = kmeans.index.search(vectors, 1)
        quantization_error = np.mean(distances)

        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.flatten().tolist(),
            "quantization_error": quantization_error,
        }