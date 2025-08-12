import numpy as np
from scipy.sparse import csr_matrix
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Compute graph Laplacian efficiently."""
        try:
            # Extract data
            data = np.array(problem["data"], dtype=np.float64)
            indices = np.array(problem["indices"], dtype=np.int32)
            indptr = np.array(problem["indptr"], dtype=np.int32)
            shape = tuple(problem["shape"])
            normed = problem["normed"]
            
            # Create sparse matrix
            graph = csr_matrix((data, indices, indptr), shape=shape)
            
            # Compute degree matrix (diagonal entries)
            degrees = np.array(graph.sum(axis=1)).flatten()
            
            if not normed:
                # Standard Laplacian: L = D - A
                # Create diagonal degree matrix in CSR format
                n = shape[0]
                diag_data = degrees
                diag_indices = np.arange(n, dtype=np.int32)
                diag_indptr = np.arange(n + 1, dtype=np.int32)
                D = csr_matrix((diag_data, diag_indices, diag_indptr), shape=shape)
                
                # L = D - A
                L = D - graph
            else:
                # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
                # Compute D^(-1/2)
                d_inv_sqrt = np.zeros_like(degrees)
                mask = degrees > 0
                d_inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
                
                # Scale A by D^(-1/2) from both sides
                # First scale rows
                scaled_data = data.copy()
                for i in range(len(indptr) - 1):
                    scaled_data[indptr[i]:indptr[i+1]] *= d_inv_sqrt[i]
                
                # Then scale columns
                for i in range(len(data)):
                    scaled_data[i] *= d_inv_sqrt[indices[i]]
                
                # Create scaled matrix
                scaled_A = csr_matrix((scaled_data, indices, indptr), shape=shape)
                
                # L = I - scaled_A
                n = shape[0]
                I_data = np.ones(n, dtype=np.float64)
                I_indices = np.arange(n, dtype=np.int32)
                I_indptr = np.arange(n + 1, dtype=np.int32)
                I = csr_matrix((I_data, I_indices, I_indptr), shape=shape)
                
                L = I - scaled_A
            
            # Ensure CSR format and eliminate zeros
            L = L.tocsr()
            L.eliminate_zeros()
            
            return {
                "laplacian": {
                    "data": L.data.tolist(),
                    "indices": L.indices.tolist(),
                    "indptr": L.indptr.tolist(),
                    "shape": L.shape
                }
            }
            
        except Exception:
            return {
                "laplacian": {
                    "data": [],
                    "indices": [],
                    "indptr": [],
                    "shape": problem.get("shape", (0, 0))
                }
            }