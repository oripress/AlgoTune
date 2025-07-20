import networkx as nx
import numpy as np
from typing import Any, List, Tuple, Dict

class Solver:
    def solve(self, problem: Dict[str, List[List[int]]], **kwargs) -> List[Tuple[int, int]]:
        """
        Solves the Maximum Common Subgraph problem by reducing it to the Maximum Clique problem.

        This implementation uses an optimized construction of the compatibility graph's
        adjacency matrix. It leverages NumPy's broadcasting on 4D arrays to avoid creating
        large intermediate matrices, leading to better memory usage and performance.

        The overall approach remains:
        1. Construct a compatibility graph where nodes are pairs of vertices (one from each
           graph G and H). An edge exists between ((i, p), (j, q)) if the mapping is
           consistent (i.e., A[i,j] == B[p,q]) and injective (i!=j, p!=q).
        2. Find the maximum clique in this compatibility graph using NetworkX.
        3. The nodes in the maximum clique correspond to the vertex mappings of the
           maximum common subgraph.
        """
        A_list = problem["A"]
        B_list = problem["B"]

        # Handle empty or malformed input graphs
        if not A_list or not B_list or not A_list[0] or not B_list[0]:
            return []

        A = np.array(A_list, dtype=bool)
        B = np.array(B_list, dtype=bool)
        n, m = A.shape[0], B.shape[0]
        
        if n == 0 or m == 0:
            return []

        # --- Optimized construction of the compatibility graph's adjacency matrix ---
        
        # Use 4D broadcasting to compute compatibility.
        # C_4d[i, p, j, q] is True if mapping i->p and j->q is compatible.
        # Compatibility means the edge relationship is preserved: A[i, j] == B[p, q].
        C_4d = (A.reshape((n, 1, n, 1)) == B.reshape((1, m, 1, m)))

        # Enforce injectivity constraints (i != j and p != q) using broadcasting.
        # This is faster and more memory-efficient than creating N*N masks.
        
        # Mask for i != j:
        # Create an (n,n) mask where diagonal is False, and broadcast it to (n,m,n,m).
        if n > 0:
            i_neq_j_mask = ~np.eye(n, dtype=bool)
            C_4d &= i_neq_j_mask.reshape((n, 1, n, 1))

        # Mask for p != q:
        # Create an (m,m) mask where diagonal is False, and broadcast it to (n,m,n,m).
        if m > 0:
            p_neq_q_mask = ~np.eye(m, dtype=bool)
            C_4d &= p_neq_q_mask.reshape((1, m, 1, m))

        # Reshape the 4D matrix into a 2D adjacency matrix for the compatibility graph.
        N = n * m
        C_adj = C_4d.reshape((N, N))
        
        # --- End of optimized construction ---

        # Create graph from the adjacency matrix.
        # networkx can handle boolean numpy arrays directly, interpreting True as an edge.
        G_comp = nx.from_numpy_array(C_adj)

        # Find the maximum clique. weight=None means unweighted (largest number of nodes).
        # nx.max_weight_clique returns a tuple (clique_nodes, weight).
        max_clique_nodes, _ = nx.max_weight_clique(G_comp, weight=None)

        # Convert clique node indices back to (graph_A_node, graph_B_node) pairs.
        if not max_clique_nodes:
            return []
        
        result = [(idx // m, idx % m) for idx in max_clique_nodes]

        return result