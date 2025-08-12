import networkx as nx
import numpy as np
from networkx.algorithms import clique

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solves the maximum clique problem using NetworkX's optimized algorithms.
        
        :param problem: A 2D adjacency matrix representing the graph.
        :return: A list of node indices that form a maximum clique.
        """
        n = len(problem)
        if n == 0:
            return []
        
        # Convert adjacency matrix to NetworkX graph
        G = nx.Graph()
        
        # Add edges based on adjacency matrix
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] == 1:
                    G.add_edge(i, j)
        
        # Find maximum clique using NetworkX
        # max_clique uses a greedy heuristic that's fast
        cliques = list(nx.find_cliques(G))
        if not cliques:
            return []
        
        # Return the largest clique
        max_clique = max(cliques, key=len)
        return sorted(list(max_clique))