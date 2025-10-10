import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Optimized graph Laplacian computation working directly with CSR format.
        """
        try:
            # Get input data - avoid unnecessary copies
            data = np.asarray(problem["data"], dtype=np.float64)
            indices = np.asarray(problem["indices"], dtype=np.int32)
            indptr = np.asarray(problem["indptr"], dtype=np.int32)
            shape = problem["shape"]
            normed = problem["normed"]
            n = shape[0]
            
            # Compute row sums (degrees) efficiently
            degrees = np.zeros(n, dtype=np.float64)
            for i in range(n):
                degrees[i] = np.sum(np.abs(data[indptr[i]:indptr[i+1]]))
            
            if not normed:
                # Standard Laplacian: L = D - A
                # Estimate output size
                nnz_est = len(data) + n
                L_data = np.zeros(nnz_est, dtype=np.float64)
                L_indices = np.zeros(nnz_est, dtype=np.int32)
                L_indptr = np.zeros(n + 1, dtype=np.int32)
                
                pos = 0
                for i in range(n):
                    L_indptr[i] = pos
                    row_start = indptr[i]
                    row_end = indptr[i + 1]
                    
                    has_diag = False
                    # Process row elements in order
                    for k in range(row_start, row_end):
                        j = indices[k]
                        if j < i:
                            L_indices[pos] = j
                            L_data[pos] = -data[k]
                            pos += 1
                        elif j == i:
                            has_diag = True
                            L_indices[pos] = i
                            L_data[pos] = degrees[i]
                            pos += 1
                        else:  # j > i
                            if not has_diag:
                                L_indices[pos] = i
                                L_data[pos] = degrees[i]
                                pos += 1
                                has_diag = True
                            L_indices[pos] = j
                            L_data[pos] = -data[k]
                            pos += 1
                    
                    # Add diagonal if not present
                    if not has_diag:
                        L_indices[pos] = i
                        L_data[pos] = degrees[i]
                        pos += 1
                
                L_indptr[n] = pos
                
                # Trim to actual size
                L_data = L_data[:pos]
                L_indices = L_indices[:pos]
                
            else:
                # Normalized Laplacian: I - D^(-1/2) A D^(-1/2)
                d_inv_sqrt = np.zeros(n, dtype=np.float64)
                nonzero_mask = degrees > 0
                d_inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(degrees[nonzero_mask])
                
                nnz_est = len(data) + n
                L_data = np.zeros(nnz_est, dtype=np.float64)
                L_indices = np.zeros(nnz_est, dtype=np.int32)
                L_indptr = np.zeros(n + 1, dtype=np.int32)
                
                pos = 0
                for i in range(n):
                    L_indptr[i] = pos
                    row_start = indptr[i]
                    row_end = indptr[i + 1]
                    
                    has_diag = False
                    for k in range(row_start, row_end):
                        j = indices[k]
                        if j < i:
                            L_indices[pos] = j
                            L_data[pos] = -d_inv_sqrt[i] * data[k] * d_inv_sqrt[j]
                            pos += 1
                        elif j == i:
                            has_diag = True
                            L_indices[pos] = i
                            L_data[pos] = 1.0 - d_inv_sqrt[i] * data[k] * d_inv_sqrt[i]
                            pos += 1
                        else:  # j > i
                            if not has_diag:
                                L_indices[pos] = i
                                L_data[pos] = 1.0
                                pos += 1
                                has_diag = True
                            L_indices[pos] = j
                            L_data[pos] = -d_inv_sqrt[i] * data[k] * d_inv_sqrt[j]
                            pos += 1
                    
                    if not has_diag:
                        L_indices[pos] = i
                        L_data[pos] = 1.0
                        pos += 1
                
                L_indptr[n] = pos
                
                L_data = L_data[:pos]
                L_indices = L_indices[:pos]
            
            return {
                "laplacian": {
                    "data": L_data.tolist(),
                    "indices": L_indices.tolist(),
                    "indptr": L_indptr.tolist(),
                    "shape": shape,
                }
            }
            
        except Exception:
            return {
                "laplacian": {
                    "data": [],
                    "indices": [],
                    "indptr": [],
                    "shape": problem.get("shape", (0, 0)),
                }
            }