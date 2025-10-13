from typing import Any, Dict, List

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.optimize import linear_sum_assignment

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Dict[str, List[int]]]:
        try:
            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            shape = problem["shape"]
            n, m = int(shape[0]), int(shape[1])
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        if n == 0 or m == 0 or n != m:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        # Convert to numpy arrays for faster ops
        try:
            indptr_arr = np.asarray(indptr, dtype=np.int64)
            indices_arr = np.asarray(indices, dtype=np.int64)
            data_arr = np.asarray(data, dtype=float)
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        nnz = int(data_arr.size)

        # Fast path: fully dense matrix in CSR with n entries per row
        try:
            if nnz == n * n and indptr_arr.size == n + 1:
                row_counts = indptr_arr[1:] - indptr_arr[:-1]
                if np.all(row_counts == n):
                    # Check if columns are [0..n-1] in each row in-order: then data is row-major dense
                    # indices should equal tile(arange(n)) length n*n
                    expected = np.tile(np.arange(n, dtype=indices_arr.dtype), n)
                    if indices_arr.shape == expected.shape and np.array_equal(indices_arr, expected):
                        # Direct reshape without further work
                        dense = data_arr.reshape(n, n)
                        row_ind, col_ind = linear_sum_assignment(dense)
                        return {
                            "assignment": {
                                "row_ind": row_ind.tolist(),
                                "col_ind": col_ind.tolist(),
                            }
                        }
                    else:
                        # Fully dense but column order per row may be permuted; build dense efficiently
                        dense = np.empty((n, n), dtype=float)
                        for r in range(n):
                            start = int(indptr_arr[r])
                            end = int(indptr_arr[r + 1])
                            dense[r, indices_arr[start:end]] = data_arr[start:end]
                        row_ind, col_ind = linear_sum_assignment(dense)
                        return {
                            "assignment": {
                                "row_ind": row_ind.tolist(),
                                "col_ind": col_ind.tolist(),
                            }
                        }
        except Exception:
            # Fall back to sparse path on any issue
            pass

        # Heuristic dense fallback for high-density matrices without zero-cost entries
        # Only if no zeros in actual data (so missing edges can be recognized as 0)
        try:
            density = nnz / float(n * n) if n > 0 else 0.0
            if density >= 0.6 and n <= 800:
                # If there are no true zero-valued costs, zeros in toarray denote missing edges
                # Note: np.asarray(data_arr == 0).any() is cheap
                if not (data_arr == 0).any():
                    # Build dense with +inf for missing edges
                    dense = np.full((n, n), np.inf, dtype=float)
                    for r in range(n):
                        start = int(indptr_arr[r])
                        end = int(indptr_arr[r + 1])
                        cols = indices_arr[start:end]
                        dense[r, cols] = data_arr[start:end]
                    row_ind, col_ind = linear_sum_assignment(dense)
                    return {
                        "assignment": {
                            "row_ind": row_ind.tolist(),
                            "col_ind": col_ind.tolist(),
                        }
                    }
        except Exception:
            # Ignore and fall back to sparse path
            pass

        # Sparse general path
        try:
            mat = sp.csr_matrix((data_arr, indices_arr, indptr_arr), shape=(n, m))
            row_ind, col_ind = min_weight_full_bipartite_matching(mat)
            return {
                "assignment": {
                    "row_ind": row_ind.tolist(),
                    "col_ind": col_ind.tolist(),
                }
            }
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}