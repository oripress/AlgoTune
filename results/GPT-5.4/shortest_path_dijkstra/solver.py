from typing import Any

import numpy as np

from apsp_cy import apsp_floyd_warshall_undirected

class Solver:
    def __init__(self) -> None:
        self._apsp = apsp_floyd_warshall_undirected

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        n = int(problem["shape"][0])
        if n == 0:
            return {"distance_matrix": []}

        data = np.asarray(problem["data"], dtype=np.float32)
        indices = np.asarray(problem["indices"], dtype=np.int32)
        indptr = np.asarray(problem["indptr"], dtype=np.int32)
        dist = self._apsp(data, indices, indptr, n)
        return {"distance_matrix": dist.tolist()}