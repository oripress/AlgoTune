import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

class Solver:
    def solve(self, problem, **kwargs):
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        if num_nodes <= 1 or not edges:
            return {"mst_edges": []}
        
        # Precompute min and max nodes
        u_arr = np.array([u for u, _, _ in edges], dtype=int)
        v_arr = np.array([v for _, v, _ in edges], dtype=int)
        weights = np.array([w for _, _, w in edges], dtype=float)
        
        # Normalize node order
        min_nodes = np.minimum(u_arr, v_arr)
        max_nodes = np.maximum(u_arr, v_arr)
        
        # Create sparse matrix directly
        graph = coo_matrix(
            (weights, (min_nodes, max_nodes)), 
            shape=(num_nodes, num_nodes)
        )
        
        # Compute MST
        mst = minimum_spanning_tree(graph)
        mst_coo = mst.tocoo()
        
        # Extract MST edges with normalized direction
        min_nodes = np.minimum(mst_coo.row, mst_coo.col)
        max_nodes = np.maximum(mst_coo.row, mst_coo.col)
        
        # Build final edge list
        mst_edges = np.column_stack([min_nodes, max_nodes, mst_coo.data])
        mst_edges = mst_edges.tolist()
        mst_edges.sort(key=lambda x: (x[0], x[1]))
        
        return {"mst_edges": mst_edges}
        # Sort by u then v
        result.sort(key=lambda x: (x[0], x[1]))
        return {"mst_edges": result}