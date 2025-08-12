import numpy as np
from numba import jit
from typing import Any

@jit(nopython=True)
def compute_laplacian_numba(data, indices, indptr, n, normed):
    # Compute degrees
    degrees = np.zeros(n)
    for i in range(n):
        row_sum = 0.0
        for j in range(indptr[i], indptr[i+1]):
            row_sum += data[j]
        degrees[i] = row_sum
    
    # For normalized Laplacian, compute inverse square roots of degrees
    inv_sqrt_degrees = np.zeros(n)
    if normed:
        for i in range(n):
            if degrees[i] == 0:
                degrees[i] = 1.0
            inv_sqrt_degrees[i] = 1.0 / np.sqrt(degrees[i])
    
    # Count total non-zeros
    total_nnz = len(data) + n
    
    # Pre-allocate result arrays
    laplacian_data = np.zeros(total_nnz)
    laplacian_indices = np.zeros(total_nnz, dtype=np.int32)
    laplacian_indptr = np.zeros(n + 1, dtype=np.int32)
    
    # Build the Laplacian
    pos = 0
    laplacian_indptr[0] = 0
    
    for i in range(n):
        # Count elements in this row
        row_start = indptr[i]
        row_end = indptr[i+1]
        row_nnz = (row_end - row_start) + 1  # +1 for diagonal
        
        # Start with diagonal
        if not normed:
            laplacian_data[pos] = degrees[i]
        else:
            laplacian_data[pos] = 1.0
        laplacian_indices[pos] = i
        pos += 1
        
        # Add off-diagonal elements
        for j in range(row_start, row_end):
            col = indices[j]
            if col != i:  # Skip self-loops
                if not normed:
                    laplacian_data[pos] = -data[j]
                else:
                    laplacian_data[pos] = -inv_sqrt_degrees[i] * data[j] * inv_sqrt_degrees[col]
                laplacian_indices[pos] = col
                pos += 1
        
        # Sort this row by column indices
        row_data = laplacian_data[laplacian_indptr[i]:pos]
        row_indices = laplacian_indices[laplacian_indptr[i]:pos]
        
        # Simple bubble sort for small arrays (numba friendly)
        for k in range(len(row_indices) - 1):
            for l in range(len(row_indices) - k - 1):
                if row_indices[l] > row_indices[l + 1]:
                    row_indices[l], row_indices[l + 1] = row_indices[l + 1], row_indices[l]
                    row_data[l], row_data[l + 1] = row_data[l + 1], row_data[l]
        
        # Copy back
        laplacian_data[laplacian_indptr[i]:pos] = row_data
        laplacian_indices[laplacian_indptr[i]:pos] = row_indices
        
        laplacian_indptr[i + 1] = pos
    
    # Trim arrays
    laplacian_data = laplacian_data[:pos]
    laplacian_indices = laplacian_indices[:pos]
    
    return laplacian_data, laplacian_indices, laplacian_indptr

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Computes the graph Laplacian using numba-optimized operations.
        
        :param problem: A dictionary representing the graph (CSR) and `normed` flag.
        :return: A dictionary with key "laplacian" containing CSR components.
        """
        try:
            # Extract input parameters
            data = np.array(problem["data"], dtype=np.float64)
            indices = np.array(problem["indices"], dtype=np.int32)
            indptr = np.array(problem["indptr"], dtype=np.int32)
            shape = tuple(problem["shape"])
            normed = problem["normed"]
            n = shape[0]
            
            # Compute Laplacian using numba
            laplacian_data, laplacian_indices, laplacian_indptr = compute_laplacian_numba(
                data, indices, indptr, n, normed
            )
            
            return {
                "laplacian": {
                    "data": laplacian_data.tolist(),
                    "indices": laplacian_indices.tolist(),
                    "indptr": laplacian_indptr.tolist(),
                    "shape": shape,
                }
            }
            
        except Exception:
            # Return empty result on failure
            return {
                "laplacian": {
                    "data": [],
                    "indices": [],
                    "indptr": [],
                    "shape": problem.get("shape", (0, 0)),
                }
            }