from typing import Any, Dict, List
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.optimize import linear_sum_assignment
from collections import deque

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Dict[str, List[int]]]:
        empty_result = {"assignment": {"row_ind": [], "col_ind": []}}

        # Basic validation
        if not isinstance(problem, dict):
            return empty_result
        for key in ("data", "indices", "indptr", "shape"):
            if key not in problem:
                return empty_result

        shape = problem["shape"]
        if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
            return empty_result
        try:
            n = int(shape[0])
            m = int(shape[1])
        except Exception:
            return empty_result
        if n != m or n < 0:
            return empty_result
        if n == 0:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        # Convert CSR arrays to numpy
        try:
            data = np.asarray(problem["data"], dtype=float)
            indices = np.asarray(problem["indices"], dtype=np.int64)
            indptr = np.asarray(problem["indptr"], dtype=np.int64)
        except Exception:
            return empty_result

        # Sanity checks for CSR structure
        if indptr.ndim != 1 or indptr.size != n + 1:
            return empty_result
        if indptr[-1] < 0 or indptr[-1] > data.size or indptr[-1] > indices.size:
            return empty_result

        # Build CSR matrix
        try:
            mat = sp.csr_matrix((data, indices, indptr), shape=(n, n))
        except Exception:
            return empty_result

        nnz = int(mat.nnz)
        density = nnz / float(max(1, n * n))

        # Quick check: no possible perfect matching if any row or column has zero degree
        row_counts = np.diff(indptr)
        if np.any(row_counts == 0):
            return empty_result
        # compute column counts efficiently
        try:
            col_counts = np.bincount(indices, minlength=n)
        except Exception:
            # fallback: build from sparse matrix
            col_counts = np.bincount(mat.tocoo().col, minlength=n)
        if np.any(col_counts == 0):
            return empty_result

        # Decide whether to attempt degree-1 elimination (peeling).
        # This is beneficial for sparse graphs with many forced matches.
        do_peel = (density <= 0.05 and n >= 80 and nnz > 0)

        assigned_rows = -np.ones(n, dtype=np.int64)
        assigned_cols = -np.ones(n, dtype=np.int64)

        if do_peel:
            try:
                coo = mat.tocoo(copy=False)
                r = coo.row.astype(np.int64)
                c = coo.col.astype(np.int64)
                d = coo.data

                nnz = r.size
                # counts
                row_counts = np.bincount(r, minlength=n)
                col_counts = np.bincount(c, minlength=n)

                # quick abort if any zero-degree (shouldn't happen due to earlier check)
                if np.any(row_counts == 0) or np.any(col_counts == 0):
                    do_peel = False
                else:
                    # sort edges by row and by col
                    e_idx = np.arange(nnz, dtype=np.int64)
                    order_by_row = np.argsort(r, kind="quicksort")
                    e_by_row = e_idx[order_by_row]
                    order_by_col = np.argsort(c, kind="quicksort")
                    e_by_col = e_idx[order_by_col]

                    row_ptr = np.empty(n + 1, dtype=np.int64)
                    col_ptr = np.empty(n + 1, dtype=np.int64)
                    row_ptr[0] = 0
                    row_ptr[1:] = np.cumsum(row_counts)
                    col_ptr[0] = 0
                    col_ptr[1:] = np.cumsum(col_counts)

                    edge_alive = np.ones(nnz, dtype=np.bool_)
                    row_alive = np.ones(n, dtype=np.bool_)
                    col_alive = np.ones(n, dtype=np.bool_)
                    row_deg = row_counts.copy()
                    col_deg = col_counts.copy()

                    q = deque()
                    for i in range(n):
                        if row_deg[i] == 1:
                            q.append(("r", i))
                        elif col_deg[i] == 1:
                            q.append(("c", i))

                    # Peel forced degree-1 vertices
                    while q:
                        typ, vid = q.popleft()
                        if typ == "r":
                            r0 = int(vid)
                            if (not row_alive[r0]) or row_deg[r0] != 1:
                                continue
                            start = row_ptr[r0]
                            end = row_ptr[r0 + 1]
                            seg = e_by_row[start:end]
                            chosen_e = -1
                            for e in seg:
                                if edge_alive[e]:
                                    chosen_e = int(e)
                                    break
                            if chosen_e == -1:
                                row_alive[r0] = False
                                continue
                            c0 = int(c[chosen_e])
                            if not col_alive[c0]:
                                row_alive[r0] = False
                                continue
                            # assign r0 -> c0
                            assigned_rows[r0] = c0
                            assigned_cols[c0] = r0
                            # remove edges of row r0
                            for e in seg:
                                if edge_alive[e]:
                                    edge_alive[e] = False
                                    col_j = int(c[e])
                                    col_deg[col_j] -= 1
                                    if col_deg[col_j] == 1 and col_alive[col_j]:
                                        q.append(("c", col_j))
                            row_alive[r0] = False
                            # remove edges of column c0
                            startc = col_ptr[c0]
                            endc = col_ptr[c0 + 1]
                            segc = e_by_col[startc:endc]
                            for e in segc:
                                if edge_alive[e]:
                                    edge_alive[e] = False
                                    row_j = int(r[e])
                                    row_deg[row_j] -= 1
                                    if row_deg[row_j] == 1 and row_alive[row_j]:
                                        q.append(("r", row_j))
                            col_alive[c0] = False

                        else:  # typ == "c"
                            c0 = int(vid)
                            if (not col_alive[c0]) or col_deg[c0] != 1:
                                continue
                            start = col_ptr[c0]
                            end = col_ptr[c0 + 1]
                            seg = e_by_col[start:end]
                            chosen_e = -1
                            for e in seg:
                                if edge_alive[e]:
                                    chosen_e = int(e)
                                    break
                            if chosen_e == -1:
                                col_alive[c0] = False
                                continue
                            r0 = int(r[chosen_e])
                            if not row_alive[r0]:
                                col_alive[c0] = False
                                continue
                            assigned_rows[r0] = c0
                            assigned_cols[c0] = r0
                            # remove edges of column c0
                            for e in seg:
                                if edge_alive[e]:
                                    edge_alive[e] = False
                                    row_j = int(r[e])
                                    row_deg[row_j] -= 1
                                    if row_deg[row_j] == 1 and row_alive[row_j]:
                                        q.append(("r", row_j))
                            col_alive[c0] = False
                            # remove edges of row r0
                            startr = row_ptr[r0]
                            endr = row_ptr[r0 + 1]
                            segr = e_by_row[startr:endr]
                            for e in segr:
                                if edge_alive[e]:
                                    edge_alive[e] = False
                                    col_j = int(c[e])
                                    col_deg[col_j] -= 1
                                    if col_deg[col_j] == 1 and col_alive[col_j]:
                                        q.append(("c", col_j))
                            row_alive[r0] = False

                    # Compute residual unassigned sets
                    unassigned_rows = np.where(assigned_rows == -1)[0]
                    unassigned_cols = np.where(assigned_cols == -1)[0]

                    # If nothing left to solve, return direct assignment
                    if unassigned_rows.size == 0:
                        row_list = [int(i) for i in range(n)]
                        col_list = [int(assigned_rows[i]) for i in range(n)]
                        return {"assignment": {"row_ind": row_list, "col_ind": col_list}}

                    # Build reduced problem from remaining alive edges
                    alive_edge_idx = np.nonzero(edge_alive)[0]
                    if alive_edge_idx.size == 0:
                        # No edges left but some vertices unassigned -> invalid
                        return empty_result

                    # Remap old indices to reduced indices
                    remap_row = -np.ones(n, dtype=np.int64)
                    remap_col = -np.ones(n, dtype=np.int64)
                    remap_row[unassigned_rows] = np.arange(unassigned_rows.size, dtype=np.int64)
                    remap_col[unassigned_cols] = np.arange(unassigned_cols.size, dtype=np.int64)

                    rem_r = r[alive_edge_idx]
                    rem_c = c[alive_edge_idx]
                    rem_d = d[alive_edge_idx]

                    # Keep only edges between unassigned rows and cols (safety)
                    mask_rows = remap_row[rem_r] >= 0
                    mask_cols = remap_col[rem_c] >= 0
                    mask_keep = mask_rows & mask_cols
                    if not np.any(mask_keep):
                        return empty_result
                    rem_r = rem_r[mask_keep]
                    rem_c = rem_c[mask_keep]
                    rem_d = rem_d[mask_keep]

                    new_rows = remap_row[rem_r]
                    new_cols = remap_col[rem_c]
                    new_n = unassigned_rows.size
                    # Build reduced CSR
                    try:
                        reduced_mat = sp.csr_matrix((rem_d, (new_rows, new_cols)), shape=(new_n, new_n))
                    except Exception:
                        # Fallback: abandon peeling
                        do_peel = False
                        reduced_mat = None
                # end else
            except Exception:
                # If anything fails, skip peeling
                do_peel = False
                reduced_mat = None
        else:
            reduced_mat = None

        # If peeling was not done or failed, treat the whole matrix as reduced
        if not do_peel or reduced_mat is None:
            reduced_mat = mat
            unassigned_rows = np.arange(n, dtype=np.int64)
            unassigned_cols = np.arange(n, dtype=np.int64)

        new_n = reduced_mat.shape[0]
        # choose solver heuristics for reduced problem
        nnz_red = int(reduced_mat.nnz)
        density_red = nnz_red / float(max(1, new_n * new_n))

        use_dense = False
        # Prefer dense Hungarian for small or moderately dense reduced problems
        if new_n <= 700:
            use_dense = True
        elif density_red >= 0.20:
            use_dense = True
        elif density_red >= 0.05 and new_n <= 3000:
            use_dense = True

        final_row_list: List[int] = []
        final_col_list: List[int] = []

        # If peeling produced partial assignment, pre-fill final lists with placeholders
        if assigned_rows is not None and np.any(assigned_rows != -1):
            # We'll fill these later
            pass

        solved_ok = False
        if use_dense:
            try:
                coo2 = reduced_mat.tocoo()
                rr = coo2.row
                cc = coo2.col
                dd = coo2.data
                dense = np.full((new_n, new_n), np.inf, dtype=float)
                if rr.size:
                    dense[rr, cc] = dd
                r_idx, c_idx = linear_sum_assignment(dense)
                # Map back to original indices
                for rr_i, cc_i in zip(r_idx.tolist(), c_idx.tolist()):
                    orig_r = int(unassigned_rows[int(rr_i)])
                    orig_c = int(unassigned_cols[int(cc_i)])
                    assigned_rows[orig_r] = orig_c
                    assigned_cols[orig_c] = orig_r
                solved_ok = True
            except MemoryError:
                solved_ok = False
            except Exception:
                solved_ok = False

        if not solved_ok:
            # Sparse fallback
            try:
                row_ind, col_ind = min_weight_full_bipartite_matching(reduced_mat)
                # Map back
                for rr_i, cc_i in zip(row_ind.tolist(), col_ind.tolist()):
                    orig_r = int(unassigned_rows[int(rr_i)])
                    orig_c = int(unassigned_cols[int(cc_i)])
                    assigned_rows[orig_r] = orig_c
                    assigned_cols[orig_c] = orig_r
                solved_ok = True
            except Exception:
                return empty_result

        # Final sanity: assigned_rows and assigned_cols must be full permutations
        if assigned_rows is None or assigned_cols is None:
            return empty_result
        if (assigned_rows.size != n) or (assigned_cols.size != n):
            return empty_result
        if np.any(assigned_rows < 0) or np.any(assigned_cols < 0):
            return empty_result
        # Ensure permutation property
        if set(assigned_rows.tolist()) != set(range(n)) or set(assigned_cols.tolist()) != set(range(n)):
            return empty_result

        row_list = [int(i) for i in range(n)]
        col_list = [int(assigned_rows[i]) for i in range(n)]
        return {"assignment": {"row_ind": row_list, "col_ind": col_list}}