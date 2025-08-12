import numpy as np
from scipy.sparse import csr_matrix, diags

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the graph Laplacian (standard or symmetric normalized) 
        from CSR-format adjacency input using fast sparse operations.
        """
        # Parse input
        try:
            data = np.array(problem["data"], dtype=float)
            indices = np.array(problem["indices"], dtype=np.int32)
            indptr = np.array(problem["indptr"], dtype=np.int32)
            shape = tuple(problem["shape"])
            problem["shape"] = shape
            n = shape[0]
            normed = bool(problem["normed"])
        except Exception:
            # On error, return empty structure
            return {
                "laplacian": {
                    "data": [],
                    "indices": [],
                    "indptr": [],
                    "shape": shape if 'shape' in locals() else (0, 0),
                }
            }

        # Build adjacency CSR
        A = csr_matrix((data, indices, indptr), shape=shape)
        # Compute degree: sum of weights per row
        if data.size and indptr.size > 1:
            deg = np.add.reduceat(data, indptr[:-1])
        else:
            deg = np.zeros(n, dtype=float)

        if not normed:
            # Standard Laplacian: L = D - A
            D = diags(deg, 0, format="csr")
            L = D - A
        else:
            # Normalized Laplacian: L = I - D^-1/2 * A * D^-1/2
            inv_sqrt = np.zeros(n, dtype=float)
            nonzero = deg > 0
            inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
            D_inv_sqrt = diags(inv_sqrt, 0, format="csr")
            M = D_inv_sqrt @ A @ D_inv_sqrt
            # Identity
            I = diags(np.ones(n, dtype=float), 0, format="csr")
            L = I - M

        # Ensure CSR, clean zeros
        L = L.tocsr()
        L.eliminate_zeros()

        # Return CSR components
        # Convert arrays to lists for safe serialization and validation
        return {
            "laplacian": {
                "data": L.data.tolist(),
                "indices": L.indices.tolist(),
                "indptr": L.indptr.tolist(),
                "shape": L.shape,
            }
        }