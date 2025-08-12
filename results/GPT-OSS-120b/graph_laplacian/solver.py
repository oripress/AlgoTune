import numpy as np
import scipy.sparse as sp

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the graph Laplacian (standard or symmetric normalized) from a CSR
        representation of an undirected weighted graph.

        Parameters
        ----------
        problem : dict
            Must contain keys:
                - "data": list or array of edge weights
                - "indices": list or array of column indices
                - "indptr": list or array of row pointer indices
                - "shape": (n, n) tuple
                - "normed": bool, False for L = D - A, True for I - D^{-1/2} A D^{-1/2}

        Returns
        -------
        dict
            {"laplacian": {"data": ..., "indices": ..., "indptr": ..., "shape": ...}}
        """
        # Validate required fields quickly; on failure return empty CSR.
        required = ("data", "indices", "indptr", "shape", "normed")
        if not all(k in problem for k in required):
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": problem.get("shape", (0, 0))}}

        try:
            # Build CSR matrix for A
            A = sp.csr_matrix((problem["data"], problem["indices"], problem["indptr"]),
                               shape=problem["shape"])
        except Exception:
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": problem["shape"]}}

        # Ensure the matrix is square
        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": problem["shape"]}}

        # Compute degree vector
        # For symmetric undirected graphs, degree = sum of weights per row
        deg = np.asarray(A.sum(axis=1)).ravel()

        if problem["normed"]:
            # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
            # Compute D^{-1/2}, handling isolated nodes (zero degree) by leaving zeros.
            with np.errstate(divide='ignore'):
                inv_sqrt_deg = 1.0 / np.sqrt(deg)
            inv_sqrt_deg[np.isinf(inv_sqrt_deg)] = 0.0

            # Form D^{-1/2} * A * D^{-1/2}
            # Efficiently scale rows and columns.
            # Row scaling:
            D_half_inv_rows = sp.diags(inv_sqrt_deg)
            # Column scaling:
            D_half_inv = sp.diags(inv_sqrt_deg)

            # Normalized adjacency
            Anorm = D_half_inv_rows @ A @ D_half_inv
            # Identity matrix in CSR
            I = sp.identity(n, format='csr')
            L = I - Anorm
        else:
            # Standard combinatorial Laplacian: L = D - A
            # D is diagonal matrix with deg on diagonal.
            D = sp.diags(deg, format='csr')
            L = D - A

        # Ensure CSR format and eliminate explicit zeros
        if not sp.isspmatrix_csr(L):
            L = L.tocsr()
        L.eliminate_zeros()

        # Prepare output
        return {
            "laplacian": {
                "data": L.data.tolist(),
                "indices": L.indices.tolist(),
                "indptr": L.indptr.tolist(),
                "shape": (n, n)
            }
        }