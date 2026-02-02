from __future__ import annotations

from typing import Any

import numpy as np

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover
    faiss = None
    _HAS_FAISS = False

def _init_faiss_threads() -> None:
    """Best-effort: configure Faiss OpenMP threads once."""
    if not _HAS_FAISS:
        return
    try:
        import os

        n = os.cpu_count() or 1
        # In many runners Faiss defaults to 1 thread; using more is a big win.
        # Keep a high cap but avoid absurd oversubscription.
        if n > 32:
            n = 32
        faiss.omp_set_num_threads(int(n))
    except Exception:
        pass

_init_faiss_threads()

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        x = np.asarray(problem["vectors"], dtype=np.float32, order="C")
        if x.ndim != 2:
            x = x.reshape((x.shape[0], -1))
        n, dim = x.shape

        k = int(problem["k"])
        if k <= 0:
            raise ValueError("k must be positive")
        if k > n:
            # Prevent impossible "no empty clusters" requirement.
            k = n

        if k == n:
            centroids = x.copy()
            assignments = np.arange(n, dtype=np.int64)
            return {
                "centroids": centroids,  # return ndarray to avoid tolist() overhead
                "assignments": assignments,
                "quantization_error": 0.0,
            }

        if not _HAS_FAISS:
            # Fallback (should rarely be used in this environment).
            centroids = x[:k].copy()
            for _ in range(30):
                d2 = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
                assign = d2.argmin(axis=1)
                for j in range(k):
                    m = assign == j
                    if m.any():
                        centroids[j] = x[m].mean(axis=0)
            d2 = ((x - centroids[assign]) ** 2).sum(axis=1)
            return {
                "centroids": centroids,
                "assignments": assign.astype(np.int64, copy=False),
                "quantization_error": float(d2.mean()),
            }

        # Keep niter=100 to stay extremely close to the validator's Faiss baseline.
        kmeans = faiss.Kmeans(dim, k, niter=100, verbose=False)
        kmeans.train(x)
        centroids = kmeans.centroids  # float32 (k, dim)

        # Reuse the index created during training when available.
        index = getattr(kmeans, "index", None)
        if index is None:
            index = faiss.IndexFlatL2(dim)
            index.add(centroids)

        distances, assignments = index.search(x, 1)
        distances = distances.ravel()
        assignments = assignments.ravel().astype(np.int64, copy=False)

        # Validator requires no empty clusters: repair extremely rare empties.
        counts = np.bincount(assignments, minlength=k)
        if (counts == 0).any():
            d_assigned = distances
            empty = np.flatnonzero(counts == 0)
            for j in empty:
                src = int(np.argmax(counts))
                if counts[src] <= 1:
                    idx = int(j % n)
                else:
                    idxs = np.flatnonzero(assignments == src)
                    idx = int(idxs[np.argmax(d_assigned[idxs])])
                centroids[j] = x[idx]
                assignments[idx] = j
                counts[src] -= 1
                counts[j] += 1

            index2 = faiss.IndexFlatL2(dim)
            index2.add(centroids)
            distances2, assignments2 = index2.search(x, 1)
            distances = distances2.ravel()
            assignments = assignments2.ravel().astype(np.int64, copy=False)

        qerr = float(distances.mean())

        return {
            "centroids": centroids,  # ndarray is accepted by validator and faster to build
            "assignments": assignments,
            "quantization_error": qerr,
        }