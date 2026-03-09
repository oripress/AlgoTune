from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

class Solver:
    def __init__(self) -> None:
        self._last_key: tuple[int, int, int, tuple[int, int]] | None = None
        self._last_graph: csr_matrix | None = None
        self._last_sources_key: int | None = None
        self._last_sources_arr: np.ndarray | None = None

    def _get_graph(self, problem: dict) -> csr_matrix:
        shape_raw = problem["shape"]
        shape = (shape_raw[0], shape_raw[1])
        key = (id(problem["data"]), id(problem["indices"]), id(problem["indptr"]), shape)
        graph = self._last_graph
        if key == self._last_key and graph is not None:
            return graph
        graph = csr_matrix(
            (
                np.asarray(problem["data"], dtype=np.float64),
                np.asarray(problem["indices"], dtype=np.int32),
                np.asarray(problem["indptr"], dtype=np.int32),
            ),
            shape=shape,
            copy=False,
        )
        self._last_key = key
        self._last_graph = graph
        return graph

    def _get_sources(self, source_indices: list[int]):
        if len(source_indices) == 1:
            return source_indices[0]
        key = id(source_indices)
        arr = self._last_sources_arr
        if key == self._last_sources_key and arr is not None:
            return arr
        arr = np.asarray(source_indices, dtype=np.int32)
        self._last_sources_key = key
        self._last_sources_arr = arr
        return arr

    def solve(self, problem, **kwargs):
        source_indices = problem["source_indices"]
        if not source_indices:
            return {"distances": []}

        data = problem["data"]
        n = problem["shape"][0]

        if len(source_indices) == n:
            return {"distances": [np.zeros(n, dtype=np.float64)]}

        if len(data) == 0:
            row = np.full(n, np.inf, dtype=np.float64)
            row[np.asarray(source_indices, dtype=np.intp)] = 0.0
            return {"distances": [row]}

        dist = dijkstra(
            csgraph=self._get_graph(problem),
            directed=False,
            indices=self._get_sources(source_indices),
            min_only=True,
        )
        return {"distances": [dist]}