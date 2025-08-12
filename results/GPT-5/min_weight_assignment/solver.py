from typing import Any, Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Dict[str, List[int]]]:
        # Parse input arrays with efficient dtypes
        try:
            shape = problem.get("shape")
            if not shape:
                return {"assignment": {"row_ind": [], "col_ind": []}}
            n = int(shape[0])

            data = np.asarray(problem.get("data", ()), dtype=np.float64)
            indices = np.asarray(problem.get("indices", ()), dtype=np.int32)
            indptr = np.asarray(problem.get("indptr", ()), dtype=np.int32)
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        if n == 0:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        # Basic sanity for CSR inputs
        if indptr.size != n + 1:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        row_counts = indptr[1:] - indptr[:-1]
        nnz = int(data.size)

        # Fast path: exactly one edge per row and columns form a permutation
        if row_counts.size == n and np.all(row_counts == 1):
            cols_single = indices[indptr[:-1]]
            # Ensure columns are a permutation of 0..n-1 (O(n) via bincount)
            if np.all(np.bincount(cols_single, minlength=n) == 1):
                row_ind = np.arange(n, dtype=np.int32)
                return {
                    "assignment": {
                        "row_ind": row_ind.tolist(),
                        "col_ind": cols_single.tolist(),
                    }
                }

        # Dense reshape fast path: fully dense in row-major CSR order
        if nnz == n * n and np.all(row_counts == n):
            try:
                # Check indptr increments are exactly n and each row's indices are 0..n-1 in order
                if indptr[0] == 0 and np.all(indptr[1:] - indptr[:-1] == n):
                    idx = indices
                    expected = np.arange(n, dtype=idx.dtype)
                    if idx.size == nnz and np.all(idx.reshape(n, n) == expected):
                        dense = data.reshape(n, n)
                        row_ind, col_ind = linear_sum_assignment(dense)
                        return {
                            "assignment": {
                                "row_ind": row_ind.tolist(),
                                "col_ind": col_ind.tolist(),
                            }
                        }
            except Exception:
                pass

        # Global density
        density = nnz / float(n * n) if n > 0 else 0.0

        # Degree-1 elimination for sparse instances to reduce problem size
        # Helps when many rows/cols have only one viable edge.
        do_reduce = False
        row_ones = None
        col_degree = None
        if (n >= 64) and (density <= 0.12):
            row_ones = np.flatnonzero(row_counts == 1)
            if row_ones.size > 0:
                do_reduce = True
            else:
                # Only compute column degrees if needed
                col_degree = np.bincount(indices, minlength=n)
                if np.any(col_degree == 1):
                    do_reduce = True

        if do_reduce:
            # Build edge-wise row indices and column-to-edge mapping
            try:
                if col_degree is None:
                    col_degree = np.bincount(indices, minlength=n)

                rows_edge = np.repeat(np.arange(n, dtype=np.int32), row_counts)
                # Group edge positions by column via argsort to avoid Python loops
                col_pos_indptr = np.concatenate(
                    ([0], np.cumsum(col_degree, dtype=np.int64))
                )
                # Sort by column; positions for col j are in this slice
                col_edge_pos = np.argsort(indices, kind="stable").astype(np.int64, copy=False)

                # Alive flags and degrees
                edge_alive = np.ones(nnz, dtype=bool)
                row_alive = np.ones(n, dtype=bool)
                col_alive = np.ones(n, dtype=bool)
                row_degree = row_counts.copy().astype(np.int64, copy=False)
                col_degree_live = col_degree.copy().astype(np.int64, copy=False)

                # Result mapping for forced assignments
                forced_col_of_row = np.full(n, -1, dtype=np.int64)

                # Initialize queues
                if row_ones is not None:
                    rows_q = list(row_ones)
                else:
                    rows_q = list(np.flatnonzero(row_degree == 1))
                cols_q = list(np.flatnonzero(col_degree_live == 1))

                # Helpers to remove rows/cols and update degrees
                def remove_row(i: int) -> None:
                    if not row_alive[i]:
                        return
                    row_alive[i] = False
                    start, end = int(indptr[i]), int(indptr[i + 1])
                    for k in range(start, end):
                        if not edge_alive[k]:
                            continue
                        edge_alive[k] = False
                        c = int(indices[k])
                        if col_alive[c]:
                            col_degree_live[c] -= 1
                            if col_degree_live[c] == 1:
                                cols_q.append(c)

                def remove_col(j: int) -> None:
                    if not col_alive[j]:
                        return
                    col_alive[j] = False
                    s, e = int(col_pos_indptr[j]), int(col_pos_indptr[j + 1])
                    # Iterate over all edges incident to column j
                    for idx_pos in range(s, e):
                        k = int(col_edge_pos[idx_pos])
                        if not edge_alive[k]:
                            continue
                        edge_alive[k] = False
                        r = int(rows_edge[k])
                        if row_alive[r]:
                            row_degree[r] -= 1
                            if row_degree[r] == 1:
                                rows_q.append(r)

                # Process queues
                while rows_q or cols_q:
                    if rows_q:
                        i = int(rows_q.pop())
                        if not row_alive[i] or row_degree[i] != 1:
                            continue
                        # Find the single alive neighbor column j for row i
                        j = -1
                        st, en = int(indptr[i]), int(indptr[i + 1])
                        for k in range(st, en):
                            if edge_alive[k]:
                                cj = int(indices[k])
                                if col_alive[cj]:
                                    j = cj
                                    break
                        if j == -1:
                            # Might have been reduced already
                            continue
                        # Assign and remove row and column
                        forced_col_of_row[i] = j
                        remove_row(i)
                        remove_col(j)
                    else:
                        j = int(cols_q.pop())
                        if not col_alive[j] or col_degree_live[j] != 1:
                            continue
                        # Find the single alive neighbor row r for column j
                        r = -1
                        s, e = int(col_pos_indptr[j]), int(col_pos_indptr[j + 1])
                        for idx_pos in range(s, e):
                            k = int(col_edge_pos[idx_pos])
                            if edge_alive[k]:
                                rr = int(rows_edge[k])
                                if row_alive[rr]:
                                    r = rr
                                    break
                        if r == -1:
                            continue
                        # Assign and remove
                        forced_col_of_row[r] = j
                        remove_row(r)
                        remove_col(j)

                # Collect remaining alive rows/cols
                rows_remain = np.flatnonzero(row_alive)
                cols_remain = np.flatnonzero(col_alive)
                m = int(rows_remain.size)

                if m == 0:
                    # All assigned by reduction
                    row_ind = np.arange(n, dtype=int)
                    col_ind = forced_col_of_row.astype(int)
                    return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}

                # Build reduced subproblem CSR using alive edges
                row_map = np.full(n, -1, dtype=np.int64)
                row_map[rows_remain] = np.arange(m, dtype=np.int64)
                col_map = np.full(n, -1, dtype=np.int64)
                col_map[cols_remain] = np.arange(m, dtype=np.int64)

                alive_mask = edge_alive
                # Filter edges of the subproblem
                sub_rows_old = rows_edge[alive_mask]
                sub_cols_old = indices[alive_mask]
                sub_data = data[alive_mask]
                sub_rows = row_map[sub_rows_old]
                sub_cols = col_map[sub_cols_old]

                sub_mat = csr_matrix((sub_data, (sub_rows, sub_cols)), shape=(m, m))

                # Solve reduced problem (choose dense vs sparse)
                sub_nnz = sub_mat.nnz
                sub_density = sub_nnz / float(m * m) if m > 0 else 0.0
                use_dense_sub = (m <= 160) or (sub_density >= 0.20 and m <= 1536)

                if use_dense_sub:
                    dense = np.full((m, m), np.inf, dtype=np.float64)
                    if sub_nnz:
                        # Fill dense by rows from CSR
                        rc = sub_mat.indptr[1:] - sub_mat.indptr[:-1]
                        rrep = np.repeat(np.arange(m, dtype=np.int64), rc)
                        dense.ravel()[rrep * np.int64(m) + sub_mat.indices.astype(np.int64, copy=False)] = sub_mat.data
                    sub_row, sub_col = linear_sum_assignment(dense)
                else:
                    sub_row, sub_col = min_weight_full_bipartite_matching(sub_mat)

                # Combine forced assignments with reduced solution
                col_ind_full = np.full(n, -1, dtype=int)
                # Forced ones
                forced_rows = np.flatnonzero(forced_col_of_row >= 0)
                col_ind_full[forced_rows] = forced_col_of_row[forced_rows].astype(int)
                # Reduced ones
                old_rows = rows_remain[sub_row]
                old_cols = cols_remain[sub_col]
                col_ind_full[old_rows.astype(int)] = old_cols.astype(int)

                row_ind_full = np.arange(n, dtype=int)
                return {
                    "assignment": {"row_ind": row_ind_full.tolist(), "col_ind": col_ind_full.tolist()}
                }
            except Exception:
                # If anything goes wrong in reduction, fall back to standard paths
                pass

        # Choose algorithm: dense Hungarian for dense/small, sparse matching otherwise
        use_dense = (n <= 160) or (density >= 0.20 and n <= 1536)

        if use_dense:
            try:
                # Construct dense with +inf for missing edges to preserve sparsity semantics
                dense = np.full((n, n), np.inf, dtype=np.float64)
                if nnz:
                    # For very sparse fills, a per-row scatter loop can be faster and lighter
                    if density < 0.08 and n > 64:
                        for i in range(n):
                            start, end = int(indptr[i]), int(indptr[i + 1])
                            if start != end:
                                cols_i = indices[start:end]
                                dense[i, cols_i] = data[start:end]
                    else:
                        # Vectorized scatter via flat indices for speed
                        rows = np.repeat(np.arange(n, dtype=np.int64), row_counts)
                        cols = indices.astype(np.int64, copy=False)
                        flat_idx = rows * np.int64(n) + cols
                        dense.ravel()[flat_idx] = data  # data already float64
                row_ind, col_ind = linear_sum_assignment(dense)
                return {
                    "assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}
                }
            except Exception:
                # Fallback to sparse method if dense path fails (e.g., memory or infeasible)
                pass

        # Sparse method: build CSR and call SciPy's solver
        try:
            mat = csr_matrix((data, indices, indptr), shape=(n, n))
            row_ind, col_ind = min_weight_full_bipartite_matching(mat)
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}