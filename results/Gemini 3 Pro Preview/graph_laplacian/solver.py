import numpy as np
from typing import Any
import fast_laplacian

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, Any]]:
        try:
            # Ensure inputs are correct types for Cython
            data = np.asarray(problem["data"], dtype=np.float64)
            indices = np.asarray(problem["indices"], dtype=np.int32)
            indptr = np.asarray(problem["indptr"], dtype=np.int32)
            shape = problem["shape"]
            n = int(shape[0])
            normed = bool(problem["normed"])
            
            l_data, l_indices, l_indptr = fast_laplacian.compute_laplacian_cython(
                data, indices, indptr, n, normed
            )
            
            return {
                "laplacian": {
                    "data": l_data.tolist(),
                    "indices": l_indices.tolist(),
                    "indptr": l_indptr.tolist(),
                    "shape": (n, n)
                }
            }
        except Exception:
            return {
                "laplacian": {
                    "data": [],
                    "indices": [],
                    "indptr": [],
                    "shape": problem.get("shape", (0, 0))
                }
            }