import numpy as np
from numba import njit

@njit(cache=True)
def _laplacian_unnormed(data, indices, indptr, n):
    # Count nnz: original off-diagonal + n diagonal entries
    # For each row, we need to insert a diagonal entry
    # New indptr, indices, data
    
    # First pass: count new nnz per row (original nnz + 1 for diagonal if not present,
    # or same if diagonal already present)
    # Actually for graph Laplacian: L = D - A
    # Diagonal of L = degree of node = sum of row weights
    # Off-diagonal of L = -A[i,j]
    
    # We need to handle the case where diagonal entries exist in A
    # For each row, the output has: all original entries negated, plus diagonal = degree
    # If diagonal already in A, we replace it; otherwise we insert it
    
    # First, compute degrees and find diagonal positions
    new_nnz = 0
    for i in range(n):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        has_diag = False
        for j in range(row_start, row_end):
            if indices[j] == i:
                has_diag = True
            new_nnz += 1
        if not has_diag:
            new_nnz += 1
    
    new_data = np.empty(new_nnz, dtype=np.float64)
    new_indices = np.empty(new_nnz, dtype=np.int32)
    new_indptr = np.empty(n + 1, dtype=np.int32)
    
    pos = 0
    for i in range(n):
        new_indptr[i] = pos
        row_start = indptr[i]
        row_end = indptr[i + 1]
        
        # Compute degree (sum of row)
        degree = 0.0
        for j in range(row_start, row_end):
            degree += data[j]
        
        # Check if diagonal exists and find its position
        has_diag = False
        diag_inserted = False
        for j in range(row_start, row_end):
            col = indices[j]
            if col == i:
                has_diag = True
                # Diagonal entry: degree - A[i,i]
                new_data[pos] = degree - data[j]
                new_indices[pos] = col
                pos += 1
            else:
                # Need to insert diagonal before this entry if col > i and not yet inserted
                if not has_diag and not diag_inserted and col > i:
                    new_data[pos] = degree
                    new_indices[pos] = i
                    pos += 1
                    diag_inserted = True
                new_data[pos] = -data[j]
                new_indices[pos] = col
                pos += 1
        
        # If diagonal wasn't in original and wasn't inserted yet (all cols < i or empty row)
        if not has_diag and not diag_inserted:
            new_data[pos] = degree
            new_indices[pos] = i
            pos += 1
    
    new_indptr[n] = pos
    return new_data[:pos], new_indices[:pos], new_indptr

@njit(cache=True)
def _laplacian_normed(data, indices, indptr, n):
    # Compute degrees
    deg = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(indptr[i], indptr[i + 1]):
            deg[i] += data[j]
    
    # Compute D^{-1/2}
    d_inv_sqrt = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if deg[i] > 0:
            d_inv_sqrt[i] = 1.0 / np.sqrt(deg[i])
    
    # Count new nnz
    new_nnz = 0
    for i in range(n):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        has_diag = False
        for j in range(row_start, row_end):
            if indices[j] == i:
                has_diag = True
            new_nnz += 1
        if not has_diag and deg[i] > 0:
            new_nnz += 1
    
    new_data = np.empty(new_nnz, dtype=np.float64)
    new_indices = np.empty(new_nnz, dtype=np.int32)
    new_indptr = np.empty(n + 1, dtype=np.int32)
    
    pos = 0
    for i in range(n):
        new_indptr[i] = pos
        row_start = indptr[i]
        row_end = indptr[i + 1]
        
        has_diag = False
        diag_inserted = False
        
        for j in range(row_start, row_end):
            col = indices[j]
            if col == i:
                has_diag = True
                # Diagonal: 1 - d_inv_sqrt[i]^2 * A[i,i] = 1 - A[i,i]/deg[i]
                if deg[i] > 0:
                    new_data[pos] = 1.0 - d_inv_sqrt[i] * data[j] * d_inv_sqrt[i]
                else:
                    new_data[pos] = 0.0
                new_indices[pos] = col
                pos += 1
            else:
                if not has_diag and not diag_inserted and col > i and deg[i] > 0:
                    new_data[pos] = 1.0
                    new_indices[pos] = i
                    pos += 1
                    diag_inserted = True
                # Off-diagonal: -d_inv_sqrt[i] * A[i,j] * d_inv_sqrt[j]
                val = -d_inv_sqrt[i] * data[j] * d_inv_sqrt[col]
                new_data[pos] = val
                new_indices[pos] = col
                pos += 1
        
        if not has_diag and not diag_inserted and deg[i] > 0:
            new_data[pos] = 1.0
            new_indices[pos] = i
            pos += 1
    
    new_indptr[n] = pos
    return new_data[:pos], new_indices[:pos], new_indptr

class Solver:
    def __init__(self):
        # Warm up numba
        d = np.array([1.0, 1.0], dtype=np.float64)
        idx = np.array([1, 0], dtype=np.int32)
        ptr = np.array([0, 1, 2], dtype=np.int32)
        _laplacian_unnormed(d, idx, ptr, 2)
        _laplacian_normed(d, idx, ptr, 2)

    def solve(self, problem, **kwargs):
        data = np.asarray(problem["data"], dtype=np.float64)
        indices = np.asarray(problem["indices"], dtype=np.int32)
        indptr = np.asarray(problem["indptr"], dtype=np.int32)
        n = problem["shape"][0]
        normed = problem["normed"]
        
        if normed:
            new_data, new_indices, new_indptr = _laplacian_normed(data, indices, indptr, n)
        else:
            new_data, new_indices, new_indptr = _laplacian_unnormed(data, indices, indptr, n)
        
        # Eliminate zeros
        mask = new_data != 0.0
        if not np.all(mask):
            # Need to rebuild without zeros
            from scipy.sparse import csr_matrix
            L = csr_matrix((new_data, new_indices, new_indptr), shape=(n, n))
            L.eliminate_zeros()
            new_data = L.data
            new_indices = L.indices
            new_indptr = L.indptr
        
        return {
            "laplacian": {
                "data": new_data,
                "indices": new_indices,
                "indptr": new_indptr,
                "shape": (n, n),
            }
        }