from __future__ import annotations

from typing import Any

import hdbscan
import numpy as np
import psutil

class Solver:
    def __init__(self) -> None:
        self._cores = max(1, int(psutil.cpu_count(logical=True) or 1))

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        x = np.asarray(problem["dataset"], dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x)

        n = x.shape[0]
        if n == 0:
            return {
                "labels": np.empty(0, dtype=np.int32),
                "probabilities": np.empty(0, dtype=np.float32),
                "cluster_persistence": np.empty(0, dtype=np.float32),
                "num_clusters": 0,
                "num_noise_points": 0,
            }

        min_cluster_size = int(problem.get("min_cluster_size", 5))
        min_samples = int(problem.get("min_samples", 3))

        dim = x.shape[1]
        if dim <= 20:
            algorithm = "boruvka_kdtree"
        elif dim <= 60:
            algorithm = "boruvka_balltree"
        else:
            algorithm = "generic"

        try:
            res = hdbscan.hdbscan(
                x,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                algorithm=algorithm,
                approx_min_span_tree=True,
                gen_min_span_tree=False,
                core_dist_n_jobs=self._cores,
            )
            labels = np.asarray(res[0], dtype=np.int32)
            probabilities = np.asarray(res[1], dtype=np.float32)
            cluster_persistence = np.asarray(res[2], dtype=np.float32)
        except Exception:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                algorithm=algorithm,
                approx_min_span_tree=True,
                gen_min_span_tree=False,
                core_dist_n_jobs=self._cores,
            )
            clusterer.fit(x)
            labels = np.asarray(clusterer.labels_, dtype=np.int32)
            probabilities = np.asarray(clusterer.probabilities_, dtype=np.float32)
            cluster_persistence = np.asarray(
                clusterer.cluster_persistence_, dtype=np.float32
            )

        non_noise = labels >= 0
        num_clusters = int(labels[non_noise].max()) + 1 if np.any(non_noise) else 0

        return {
            "labels": labels,
            "probabilities": probabilities,
            "cluster_persistence": cluster_persistence,
            "num_clusters": num_clusters,
            "num_noise_points": int(np.count_nonzero(labels == -1)),
        }