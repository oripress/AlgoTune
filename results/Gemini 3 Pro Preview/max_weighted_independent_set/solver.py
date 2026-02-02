import networkx as nx
import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> list:
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(adj_matrix)
        
        # Build the complement graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        # Set weights
        for i in range(n):
            G.nodes[i]['weight'] = weights[i]
        
        # Edges of the complement graph
        # Edge (i, j) exists in complement if adj_matrix[i][j] == 0 and i != j
        adj_np = np.array(adj_matrix)
        
        # We want indices where adj_np[i, j] == 0 and i < j
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        complement_edges_mask = mask & (adj_np == 0)
        
        rows, cols = np.nonzero(complement_edges_mask)
        # Using zip is faster than column_stack for iteration usually
        G.add_edges_from(zip(rows, cols))
        
        # Find max weight clique in complement graph
        clique, _ = nx.max_weight_clique(G, weight='weight')
        
        return sorted(clique)