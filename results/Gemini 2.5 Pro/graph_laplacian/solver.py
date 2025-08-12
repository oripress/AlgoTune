import numpy as np
import numba
from typing import Any

@numba.njit(cache=True, fastmath=True)
def combinatorial_laplacian_kernel(n, A_data, A_indices, A_indptr):
    """
    Computes the combinatorial graph Laplacian L = D - A.
    """
    # 1. Calculate degrees
    degrees = np.zeros(n, dtype=A_data.dtype)
    for i in range(n):
        for k in range(A_indptr[i], A_indptr[i+1]):
            degrees[i] += A_data[k]

    # 2. Allocate space for L's CSR arrays. Max possible size is nnz(A) + n.
    L_data = np.empty(len(A_data) + n, dtype=A_data.dtype)
    L_indices = np.empty(len(A_data) + n, dtype=A_indices.dtype)
    L_indptr = np.empty(n + 1, dtype=A_indptr.dtype)
    
    nnz = 0
    L_indptr[0] = 0
    for i in range(n):
        # Process row i
        row_start_A = A_indptr[i]
        row_end_A = A_indptr[i+1]
        
        diag_val = degrees[i]
        a_ptr = row_start_A
        
        # Copy elements from A before diagonal, negating them
        while a_ptr < row_end_A and A_indices[a_ptr] < i:
            L_indices[nnz] = A_indices[a_ptr]
            L_data[nnz] = -A_data[a_ptr]
            nnz += 1
            a_ptr += 1
        
        # Check for A_ii and subtract it from the degree for L_ii
        if a_ptr < row_end_A and A_indices[a_ptr] == i:
            diag_val -= A_data[a_ptr]
            a_ptr += 1
        
        # Insert diagonal for L, if non-zero
        if diag_val != 0.0:
            L_indices[nnz] = i
            L_data[nnz] = diag_val
            nnz += 1
            
        # Copy elements from A after diagonal, negating them
        while a_ptr < row_end_A:
            L_indices[nnz] = A_indices[a_ptr]
            L_data[nnz] = -A_data[a_ptr]
            nnz += 1
            a_ptr += 1
            
        L_indptr[i+1] = nnz
        
    return L_data[:nnz], L_indices[:nnz], L_indptr

@numba.njit(cache=True, fastmath=True)
def normalized_laplacian_kernel(n, A_data, A_indices, A_indptr):
    """
    Computes the normalized Laplacian L = I - D^-1/2 * A * D^-1/2.
    """
    # 1. Calculate degrees
    degrees = np.zeros(n, dtype=A_data.dtype)
    for i in range(n):
        for k in range(A_indptr[i], A_indptr[i+1]):
            degrees[i] += A_data[k]

    # 2. Calculate D^(-1/2)
    inv_sqrt_degrees = np.zeros(n, dtype=A_data.dtype)
    for i in range(n):
        if degrees[i] > 0:
            inv_sqrt_degrees[i] = 1.0 / np.sqrt(degrees[i])

    # 3. Allocate space for L's CSR arrays
    L_data = np.empty(len(A_data) + n, dtype=A_data.dtype)
    L_indices = np.empty(len(A_data) + n, dtype=A_indices.dtype)
    L_indptr = np.empty(n + 1, dtype=A_indptr.dtype)
    
    nnz = 0
    L_indptr[0] = 0
    for i in range(n):
        row_start_A = A_indptr[i]
        row_end_A = A_indptr[i+1]
        
        inv_sqrt_d_i = inv_sqrt_degrees[i]
        diag_val = 1.0
        if degrees[i] == 0:
            diag_val = 0.0
        
        a_ptr = row_start_A
        
        # Handle elements before the diagonal
        while a_ptr < row_end_A and A_indices[a_ptr] < i:
            j = A_indices[a_ptr]
            inv_sqrt_d_j = inv_sqrt_degrees[j]
            L_indices[nnz] = j
            L_data[nnz] = -A_data[a_ptr] * inv_sqrt_d_i * inv_sqrt_d_j
            nnz += 1
            a_ptr += 1
        
        # Handle the diagonal element
        if a_ptr < row_end_A and A_indices[a_ptr] == i:
            if degrees[i] > 0:
                diag_val -= A_data[a_ptr] / degrees[i]
            a_ptr += 1
        
        if diag_val != 0.0:
            L_indices[nnz] = i
            L_data[nnz] = diag_val
            nnz += 1
            
        # Handle elements after the diagonal
        while a_ptr < row_end_A:
            j = A_indices[a_ptr]
            inv_sqrt_d_j = inv_sqrt_degrees[j]
            L_indices[nnz] = j
            L_data[nnz] = -A_data[a_ptr] * inv_sqrt_d_i * inv_sqrt_d_j
            nnz += 1
            a_ptr += 1
            
        L_indptr[i+1] = nnz
        
    return L_data[:nnz], L_indices[:nnz], L_indptr

class Solver:
    def solve(self, problem: dict[str, Any]) -> Any:
        """
        Computes the graph Laplacian by dispatching to a Numba-jitted kernel.
        """
        try:
            shape = problem["shape"]
            n = shape[0]
            # Ensure input data are numpy arrays with correct types for Numba
            A_data = np.array(problem["data"], dtype=np.float64)
            A_indices = np.array(problem["indices"], dtype=np.int32)
            A_indptr = np.array(problem["indptr"], dtype=np.int32)
            normed = problem["normed"]
        except (KeyError, IndexError, TypeError):
            # Return empty valid structure on input error
            shape = problem.get("shape", (0, 0))
            n_rows = shape[0] if isinstance(shape, tuple) and len(shape) == 2 else 0
            return {
                "laplacian": {
                    "data": [],
                    "indices": [],
                    "indptr": [0] * (n_rows + 1),
                    "shape": shape,
                }
            }

        if n == 0:
            return {
                "laplacian": {
                    "data": [],
                    "indices": [],
                    "indptr": [0],
                    "shape": (0, 0),
                }
            }

        if normed:
            L_data, L_indices, L_indptr = normalized_laplacian_kernel(
                n, A_data, A_indices, A_indptr
            )
        else:
            L_data, L_indices, L_indptr = combinatorial_laplacian_kernel(
                n, A_data, A_indices, A_indptr
            )
            
        return {
            "laplacian": {
                "data": L_data.tolist(),
                "indices": L_indices.tolist(),
                "indptr": L_indptr.tolist(),
                "shape": problem["shape"],
            }
        }