import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        # Load CSR components
        data = np.array(problem["data"], dtype=np.float64)
        indices = np.array(problem["indices"], dtype=np.int64)
        indptr = np.array(problem["indptr"], dtype=np.int64)
        # Number of nodes
        n = int(problem["shape"][0])
        # Compute degrees (sum of each row)
        if data.size > 0:
            deg = np.add.reduceat(data, indptr[:-1])
        else:
            deg = np.zeros(n, dtype=np.float64)
        normed = bool(problem.get("normed", False))
        # Prepare row indices for off-diagonal entries
        row_counts = np.diff(indptr)
        row = np.repeat(np.arange(n, dtype=np.int64), row_counts)
        # Build off-diagonal and diagonal values
        if normed:
            inv_sqrt = np.zeros(n, dtype=np.float64)
            nz = deg > 0
            inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
            off_data = -data * inv_sqrt[row] * inv_sqrt[indices]
            diag_data = np.ones(n, dtype=np.float64)
        else:
            off_data = -data
            diag_data = deg
        # Combine data and indices
        data_all = np.concatenate((off_data, diag_data))
        col_all = np.concatenate((indices, np.arange(n, dtype=np.int64)))
        row_all = np.concatenate((row, np.arange(n, dtype=np.int64)))
        # Sort by row then column index
        # lexsort keys: (column, row) => primary=row, secondary=column
        perm = np.lexsort((col_all, row_all))
        dataL = data_all[perm]
        colL = col_all[perm]
        rowL = row_all[perm]
        # Build new indptr by counting entries per row
        counts = np.bincount(rowL, minlength=n)
        indptrL = np.empty(n + 1, dtype=np.int64)
        indptrL[0] = 0
        indptrL[1:] = np.cumsum(counts, dtype=np.int64)
        # Return CSR components
        return {
            "laplacian": {
                "data": dataL.tolist(),
                "indices": colL,
                "indptr": indptrL,
                "shape": (n, n),
            }
        }