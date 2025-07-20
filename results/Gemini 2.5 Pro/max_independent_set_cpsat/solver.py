import networkx as nx
import numpy as np
from typing import Any

class Solver:
    """
    This class solves the Maximum Independent Set problem by leveraging the
    highly optimized algorithms available in the networkx library.
    """

    def solve(self, problem: list[list[int]], **kwargs) -> list[int]:
        """
        Solves the Maximum Independent Set problem.

        This implementation uses the networkx library. The strategy is based on
        the fundamental graph theory equivalence:
        The Maximum Independent Set of a graph G is the Maximum Clique of its
        complement graph G'.

        Steps:
        1. Create a networkx Graph object from the input adjacency matrix.
        2. Compute the complement of this graph.
        3. Find the maximum clique in the complement graph using networkx's
           optimized `max_weight_clique` algorithm. This is much faster
           than finding all maximal cliques and then the largest one.
        4. Sort and return the resulting list of nodes.

        Args:
            problem: An adjacency matrix representation of the graph.
                     problem[i][j] == 1 if an edge exists between i and j.

        Returns:
            A sorted list of nodes in the maximum independent set.
        """
        # Convert the adjacency matrix (list of lists) to a NumPy array.
        adj_matrix = np.array(problem)

        # Create a graph from the NumPy adjacency matrix.
        G = nx.from_numpy_array(adj_matrix)

        # Handle the edge case of an empty graph.
        if G.number_of_nodes() == 0:
            return []

        # The maximum independent set of G is the maximum clique of the complement of G.
        complement_G = nx.complement(G)
        
        # Use max_weight_clique (with weight=None) to find the maximum clique.
        # This is more efficient than find_cliques() as it doesn't enumerate all
        # maximal cliques, avoiding a potential exponential number of them.
        # It returns a (clique, weight) tuple. We only need the clique.
        max_clique, _ = nx.max_weight_clique(complement_G, weight=None)

        # The result is a list of nodes, sort it as required by the problem spec.
        result = sorted(max_clique)

        return result