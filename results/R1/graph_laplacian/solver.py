import numpy as np
import numba
import scipy.sparse
import scipy.sparse.csgraph

@numba.jit(nopython=True, cache=True)
def compute_degrees(data, indices, indptr, n):
    degrees = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = indptr[i]
        end = indptr[i+1]
        for j in range(start, end):
            degrees[i] += data[j]
    return degrees

@numba.jit(nopython=True, cache=True)
def build_laplacian(data, indices, indptr, degrees, n, normed):
    # Precompute row sizes and self-loop flags
    row_sizes = np.zeros(n, dtype=np.int32)
    has_diag = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        start = indptr[i]
        end = indptr[i+1]
        for j in range(start, end):
            if indices[j] == i:
                has_diag[i] = True
        row_sizes[i] = (end - start) + (0 if has_diag[i] else 1)
    
    # Create indptr array
    new_indptr = np.zeros(n+1, dtype=np.int32)
    for i in range(1, n+1):
        new_indptr[i] = new_indptr[i-1] + row_sizes[i-1]
    total_nnz = new_indptr[-1]
    
    # Create data and indices arrays
    new_data = np.zeros(total_nnz, dtype=np.float64)
    new_indices = np.zeros(total_nnz, dtype=np.int32)
    
    # Fill data and indices
    for i in range(n):
        start_idx = new_indptr[i]
        row_size = row_sizes[i]
        if row_size == 0:
            continue
        
        # Create temporary storage for row
        temp_indices = np.zeros(row_size, dtype=np.int32)
        temp_data = np.zeros(row_size, dtype=np.float64)
        pos = 0
        
        # Process row
        row_start = indptr[i]
        row_end = indptr[i+1]
        diag_val = degrees[i]
        
        for j in range(row_start, row_end):
            col = indices[j]
            w = data[j]
            if col == i:
                diag_val -= w
            else:
                if normed:
                    # For normalized Laplacian: -w / sqrt(deg_i * deg_j)
                    val = -w / (np.sqrt(degrees[i] * degrees[col]))
                else:
                    val = -w
                temp_indices[pos] = col
                temp_data[pos] = val
                pos += 1
        
        # Add diagonal element
        if normed:
            diag_val = 1.0
        temp_indices[pos] = i
        temp_data[pos] = diag_val
        pos += 1
        
        # Sort row by column index using insertion sort
        for k in range(1, row_size):
            key_idx = temp_indices[k]
            key_val = temp_data[k]
            j = k - 1
            while j >= 0 and temp_indices[j] > key_idx:
                temp_indices[j+1] = temp_indices[j]
                temp_data[j+1] = temp_data[j]
                j -= 1
            temp_indices[j+1] = key_idx
            temp_data[j+1] = key_val
        
        # Store sorted row
        for idx in range(row_size):
            new_indices[start_idx + idx] = temp_indices[idx]
            new_data[start_idx + idx] = temp_data[idx]
    
    return new_data, new_indices, new_indptr

class Solver:
    def solve(self, problem, **kwargs):
        try:
            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            shape = problem["shape"]
            n = shape[0]
            normed = problem["normed"]
        except Exception:
            return {
                "laplacian": {
                    "data": [],
                    "indices": [],
                    "indptr": [],
                    "shape": problem.get("shape", [0, 0]),
                }
            }
        
        try:
            # Convert to arrays for numba
            data_arr = np.array(data, dtype=np.float64)
            indices_arr = np.array(indices, dtype=np.int32)
            indptr_arr = np.array(indptr, dtype=np.int32)
            
            # Compute degrees
            degrees = compute_degrees(data_arr, indices_arr, indptr_arr, n)
            
            # Build Laplacian
            L_data, L_indices, L_indptr = build_laplacian(
                data_arr, indices_arr, indptr_arr, degrees, n, normed
            )
            
            return {
                "laplacian": {
                    "data": L_data.tolist(),
                    "indices": L_indices.tolist(),
                    "indptr": L_indptr.tolist(),
                    "shape": shape,
                }
            }
        except Exception as e:
            # Fallback to scipy
            try:
                graph_csr = scipy.sparse.csr_matrix(
                    (data, indices, indptr), shape=shape
                )
                L = scipy.sparse.csgraph.laplacian(graph_csr, normed=normed)
                if not isinstance(L, scipy.sparse.csr_matrix):
                    L_csr = L.tocsr()
                else:
                    L_csr = L
                L_csr.eliminate_zeros()
                
                return {
                    "laplacian": {
                        "data": L_csr.data.tolist(),
                        "indices": L_csr.indices.tolist(),
                        "indptr": L_csr.indptr.tolist(),
                        "shape": list(L_csr.shape),
                    }
                }
            except Exception:
                return {
                    "laplacian": {
                        "data": [],
                        "indices": [],
                        "indptr": [],
                        "shape": shape,
                    }
                }