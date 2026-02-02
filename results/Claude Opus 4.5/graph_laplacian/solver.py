import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def compute_degrees(data, indptr, n):
    degrees = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for k in range(indptr[i], indptr[i+1]):
            degrees[i] += data[k]
    return degrees

@njit(cache=True)
def laplacian_unnormed(data, indices, indptr, n, degrees):
    nnz = len(data)
    for i in range(n):
        if degrees[i] != 0.0:
            has_diag = False
            for k in range(indptr[i], indptr[i+1]):
                if indices[k] == i:
                    has_diag = True
                    break
            if not has_diag:
                nnz += 1
    
    new_data = np.empty(nnz, dtype=np.float64)
    new_indices = np.empty(nnz, dtype=np.int32)
    new_indptr = np.zeros(n+1, dtype=np.int32)
    
    ptr = 0
    for i in range(n):
        new_indptr[i] = ptr
        start = indptr[i]
        end = indptr[i+1]
        
        diag_pos = -1
        for k in range(start, end):
            if indices[k] == i:
                diag_pos = k
                break
        
        diag_inserted = False
        for k in range(start, end):
            j = indices[k]
            
            if not diag_inserted and diag_pos == -1 and degrees[i] != 0.0 and j > i:
                new_indices[ptr] = i
                new_data[ptr] = degrees[i]
                ptr += 1
                diag_inserted = True
            
            if j == i:
                val = degrees[i] - data[k]
                if val != 0.0:
                    new_indices[ptr] = j
                    new_data[ptr] = val
                    ptr += 1
                diag_inserted = True
            else:
                new_indices[ptr] = j
                new_data[ptr] = -data[k]
                ptr += 1
        
        if not diag_inserted and diag_pos == -1 and degrees[i] != 0.0:
            new_indices[ptr] = i
            new_data[ptr] = degrees[i]
            ptr += 1
    
    new_indptr[n] = ptr
    return new_data[:ptr], new_indices[:ptr], new_indptr

@njit(cache=True)
def laplacian_normed(data, indices, indptr, n, degrees):
    d_inv_sqrt = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if degrees[i] > 0:
            d_inv_sqrt[i] = 1.0 / np.sqrt(degrees[i])
    
    nnz = len(data)
    for i in range(n):
        if degrees[i] > 0:
            has_diag = False
            for k in range(indptr[i], indptr[i+1]):
                if indices[k] == i:
                    has_diag = True
                    break
            if not has_diag:
                nnz += 1
    
    new_data = np.empty(nnz, dtype=np.float64)
    new_indices = np.empty(nnz, dtype=np.int32)
    new_indptr = np.zeros(n+1, dtype=np.int32)
    
    ptr = 0
    for i in range(n):
        new_indptr[i] = ptr
        start = indptr[i]
        end = indptr[i+1]
        
        diag_pos = -1
        for k in range(start, end):
            if indices[k] == i:
                diag_pos = k
                break
        
        diag_inserted = False
        for k in range(start, end):
            j = indices[k]
            
            if not diag_inserted and diag_pos == -1 and degrees[i] > 0 and j > i:
                new_indices[ptr] = i
                new_data[ptr] = 1.0
                ptr += 1
                diag_inserted = True
            
            if j == i:
                if degrees[i] > 0:
                    val = 1.0 - data[k] * d_inv_sqrt[i] * d_inv_sqrt[i]
                    if val != 0.0:
                        new_indices[ptr] = j
                        new_data[ptr] = val
                        ptr += 1
                diag_inserted = True
            else:
                val = -data[k] * d_inv_sqrt[i] * d_inv_sqrt[j]
                if val != 0.0:
                    new_indices[ptr] = j
                    new_data[ptr] = val
                    ptr += 1
        
        if not diag_inserted and diag_pos == -1 and degrees[i] > 0:
            new_indices[ptr] = i
            new_data[ptr] = 1.0
            ptr += 1
    
    new_indptr[n] = ptr
    return new_data[:ptr], new_indices[:ptr], new_indptr

class Solver:
    def solve(self, problem, **kwargs):
        data = np.asarray(problem["data"], dtype=np.float64)
        indices = np.asarray(problem["indices"], dtype=np.int32)
        indptr = np.asarray(problem["indptr"], dtype=np.int32)
        shape = tuple(problem["shape"])
        normed = problem["normed"]
        n = shape[0]
        
        if len(data) == 0:
            return {"laplacian": {"data": np.array([]), "indices": np.array([], dtype=np.int32), "indptr": np.zeros(n+1, dtype=np.int32), "shape": shape}}
        
        degrees = compute_degrees(data, indptr, n)
        
        if normed:
            new_data, new_indices, new_indptr = laplacian_normed(data, indices, indptr, n, degrees)
        else:
            new_data, new_indices, new_indptr = laplacian_unnormed(data, indices, indptr, n, degrees)
        
        return {"laplacian": {"data": new_data, "indices": new_indices, "indptr": new_indptr, "shape": shape}}