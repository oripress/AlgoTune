import networkx as nx
import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Solves the MWIS problem by finding the maximum weighted clique
        in the complement graph using networkx.
        """
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(adj_matrix)

        if n == 0:
            return []

        # Create complement graph using numpy for speed
        adj_numpy = np.array(adj_matrix)
        complement_adj = 1 - adj_numpy
        np.fill_diagonal(complement_adj, 0)
        
        G_complement = nx.from_numpy_array(complement_adj)
        
        # Add weights to the nodes efficiently
        weight_dict = {i: weights[i] for i in range(n)}
        nx.set_node_attributes(G_complement, weight_dict, 'weight')

        # Find the maximum weight clique in the complement graph.
        # This is equivalent to the max weight independent set in the original graph.
        clique, weight = nx.max_weight_clique(G_complement, weight="weight")
        
        # The empty set is a valid independent set with weight 0.
        # If the best solution found has a non-positive weight, the empty set is better or equal.
        if weight > 0:
            return sorted(list(clique))
        else:
            return []