import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

class Solver:
    def solve(self, problem, **kwargs):
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        if num_nodes <= 1 or len(edges) == 0:
            return {"mst_edges": []}
        
        # Convert edges to numpy arrays efficiently
        if hasattr(edges, '__array__'):
            edge_arr = np.asarray(edges)
        else:
            edge_arr = np.array(edges)
        
        if len(edge_arr) == 0:
            return {"mst_edges": []}
        
        u = edge_arr[:, 0].astype(np.int32)
        v = edge_arr[:, 1].astype(np.int32)
        w = edge_arr[:, 2]
        
        # Build sparse matrix (symmetric - use both directions)
        # scipy MST works on the upper triangle, so we need full matrix
        row = np.concatenate([u, v])
        col = np.concatenate([v, u])
        data = np.concatenate([w, w])
        
        graph = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        
        # Compute MST using scipy (Kruskal's in C)
        mst = minimum_spanning_tree(graph)
        
        # Extract edges from the sparse result
        mst_coo = mst.tocoo()
        rows = mst_coo.row
        cols = mst_coo.col
        weights = mst_coo.data
        
        # Ensure u < v and build result
        mask = rows > cols
        rows_fixed = np.where(mask, cols, rows)
        cols_fixed = np.where(mask, rows, cols)
        
        # Sort by (u, v)
        order = np.lexsort((cols_fixed, rows_fixed))
        
        mst_edges = []
        for i in order:
            mst_edges.append([int(rows_fixed[i]), int(cols_fixed[i]), float(weights[i])])
        
        return {"mst_edges": mst_edges}