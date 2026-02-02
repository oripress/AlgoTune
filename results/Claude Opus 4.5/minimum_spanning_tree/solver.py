import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

class Solver:
    def solve(self, problem, **kwargs):
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        if num_nodes == 0 or len(edges) == 0:
            return {"mst_edges": []}
        
        # Convert edges to numpy array for faster processing
        if isinstance(edges, np.ndarray):
            edges_arr = edges
        else:
            edges_arr = np.asarray(edges)
        
        # Build sparse matrix for scipy
        row = edges_arr[:, 0].astype(np.int32)
        col = edges_arr[:, 1].astype(np.int32)
        data = edges_arr[:, 2]
        
        # Create symmetric adjacency matrix (undirected graph)
        # Stack instead of concatenate is slightly faster
        row_full = np.concatenate((row, col))
        col_full = np.concatenate((col, row))
        data_full = np.concatenate((data, data))
        
        graph = csr_matrix((data_full, (row_full, col_full)), shape=(num_nodes, num_nodes))
        
        # Compute MST using scipy
        mst = minimum_spanning_tree(graph)
        
        # Extract edges from MST result directly from csr format
        mst_coo = mst.tocoo()
        
        u = mst_coo.row
        v = mst_coo.col
        w = mst_coo.data
        n_edges = len(w)
        
        if n_edges == 0:
            return {"mst_edges": []}
        
        # Ensure u < v for sorting consistency - vectorized
        swap_mask = u > v
        u_new = np.where(swap_mask, v, u)
        v_new = np.where(swap_mask, u, v)
        
        # Sort by (u, v) using lexsort (sorts by last key first, so v then u)
        sort_idx = np.lexsort((v_new, u_new))
        
        # Direct array indexing is faster
        u_s = u_new[sort_idx]
        v_s = v_new[sort_idx]
        w_s = w[sort_idx]
        
        # Convert to list of lists - use tolist() for speed
        u_list = u_s.tolist()
        v_list = v_s.tolist()
        w_list = w_s.tolist()
        
        mst_edges = [[u_list[i], v_list[i], w_list[i]] for i in range(n_edges)]
        
        return {"mst_edges": mst_edges}