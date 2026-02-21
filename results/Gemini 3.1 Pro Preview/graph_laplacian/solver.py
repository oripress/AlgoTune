import numpy as np
from typing import Any
from numba import njit

@njit(cache=True)
def compute_laplacian_unnormed(data, indices, indptr, n):
    degrees = np.zeros(n, dtype=data.dtype)
    for i in range(n):
        row_sum = 0.0
        for j in range(indptr[i], indptr[i+1]):
            if i != indices[j]:
                row_sum += data[j]
        degrees[i] = row_sum
    
    nnz_L_max = len(data) + n
    
    L_data = np.empty(nnz_L_max, dtype=data.dtype)
    L_indices = np.empty(nnz_L_max, dtype=indices.dtype)
    L_indptr = np.empty(n + 1, dtype=indptr.dtype)
    
    L_indptr[0] = 0
    idx = 0
    
    for i in range(n):
        deg_i = degrees[i]
        j = indptr[i]
        end = indptr[i+1]
        
        while j < end and indices[j] < i:
            val = data[j]
            if val != 0:
                L_data[idx] = -val
                L_indices[idx] = indices[j]
                idx += 1
            j += 1
            
        if deg_i != 0:
            L_data[idx] = deg_i
            L_indices[idx] = i
            idx += 1
            
        while j < end:
            col = indices[j]
            if col != i:
                val = data[j]
                if val != 0:
                    L_data[idx] = -val
                    L_indices[idx] = col
                    idx += 1
            j += 1
            
        L_indptr[i+1] = idx
        
    return L_data[:idx], L_indices[:idx], L_indptr

@njit(cache=True)
def compute_laplacian_normed(data, indices, indptr, n):
    degrees = np.zeros(n, dtype=data.dtype)
    for i in range(n):
        row_sum = 0.0
        for j in range(indptr[i], indptr[i+1]):
            if i != indices[j]:
                row_sum += data[j]
        degrees[i] = row_sum
                
    inv_sqrt_deg = np.zeros(n, dtype=data.dtype)
    for i in range(n):
        if degrees[i] > 0:
            inv_sqrt_deg[i] = 1.0 / np.sqrt(degrees[i])
            
    nnz_L_max = len(data) + n
    
    L_data = np.empty(nnz_L_max, dtype=data.dtype)
    L_indices = np.empty(nnz_L_max, dtype=indices.dtype)
    L_indptr = np.empty(n + 1, dtype=indptr.dtype)
    
    L_indptr[0] = 0
    idx = 0
    
    for i in range(n):
        diag_added = False
        deg_i = degrees[i]
        inv_sq_i = inv_sqrt_deg[i]
        diag_val = 1.0 if deg_i > 0 else 0.0
        
        for j in range(indptr[i], indptr[i+1]):
            col = indices[j]
            val = data[j]
            
            if col == i:
                if diag_val != 0:
                    L_data[idx] = diag_val
                    L_indices[idx] = i
                    idx += 1
                diag_added = True
            elif col > i and not diag_added:
                if diag_val != 0:
                    L_data[idx] = diag_val
                    L_indices[idx] = i
                    idx += 1
                diag_added = True
                
                off_val = -val * inv_sq_i * inv_sqrt_deg[col]
                if off_val != 0:
                    L_data[idx] = off_val
                    L_indices[idx] = col
                    idx += 1
            else:
                off_val = -val * inv_sq_i * inv_sqrt_deg[col]
                if off_val != 0:
                    L_data[idx] = off_val
                    L_indices[idx] = col
                    idx += 1
                
        if not diag_added:
            if diag_val != 0:
                L_data[idx] = diag_val
                L_indices[idx] = i
                idx += 1
            
        L_indptr[i+1] = idx
        
    return L_data[:idx], L_indices[:idx], L_indptr

class Solver:
    def __init__(self):
        # Trigger Numba compilation
        d = np.array([1.0], dtype=np.float64)
        i = np.array([0], dtype=np.int32)
        p = np.array([0, 1], dtype=np.int32)
        compute_laplacian_unnormed(d, i, p, 1)
        compute_laplacian_normed(d, i, p, 1)

    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, Any]]:
        try:
            data = np.asarray(problem["data"], dtype=np.float64)
            indices = np.asarray(problem["indices"], dtype=np.int32)
            indptr = np.asarray(problem["indptr"], dtype=np.int32)
            n = problem["shape"][0]
            normed = problem["normed"]
            
            if normed:
                L_data, L_indices, L_indptr = compute_laplacian_normed(data, indices, indptr, n)
            else:
                L_data, L_indices, L_indptr = compute_laplacian_unnormed(data, indices, indptr, n)
                
            return {
                "laplacian": {
                    "data": L_data,
                    "indices": L_indices,
                    "indptr": L_indptr,
                    "shape": (n, n),
                }
            }
        except Exception as e:
            return {
                "laplacian": {
                    "data": [],
                    "indices": [],
                    "indptr": [],
                    "shape": problem.get("shape", (0, 0)),
                }
            }