import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def solve(self, problem):
        """Compute the graph Laplacian efficiently."""
        try:
            # Extract CSR components
            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            shape = problem["shape"]
            normed = problem["normed"]
            
            # Create the sparse matrix
            graph_csr = scipy.sparse.csr_matrix(
                (data, indices, indptr), shape=shape
            )
            
            # Compute Laplacian using scipy
            L = scipy.sparse.csgraph.laplacian(graph_csr, normed=normed)
            
            # Ensure output is CSR format
            if not isinstance(L, scipy.sparse.csr_matrix):
                L_csr = L.tocsr()
            else:
                L_csr = L
            L_csr.eliminate_zeros()
            
            # Convert to lists to avoid numpy array issues in validation
            return {
                "laplacian": {
                    "data": L_csr.data.tolist() if hasattr(L_csr.data, 'tolist') else list(L_csr.data),
                    "indices": L_csr.indices.tolist() if hasattr(L_csr.indices, 'tolist') else list(L_csr.indices),
                    "indptr": L_csr.indptr.tolist() if hasattr(L_csr.indptr, 'tolist') else list(L_csr.indptr),
                    "shape": L_csr.shape
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