import networkx as nx
from itertools import combinations
import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> list[tuple[int, int]]:
        A = problem["A"]
        B = problem["B"]
        
        # Convert adjacency matrices to numpy arrays then to networkx graphs
        G = nx.from_numpy_array(np.array(A))
        H = nx.from_numpy_array(np.array(B))
        
        # Use networkx's built-in function for maximum common subgraph
        # This uses the VF2 algorithm which is efficient for graph isomorphism problems
        matcher = nx.algorithms.isomorphism.GraphMatcher(G, H)
        
        best_match = []
        # Try to find the largest common subgraph
        for size in range(min(len(G), len(H)), 0, -1):
            # Generate all possible subgraphs of G of size 'size'
            for nodes_G in combinations(G.nodes(), size):
                subgraph_G = G.subgraph(nodes_G)
                
                # Try to find isomorphism between this subgraph and any subgraph of H
                for nodes_H in combinations(H.nodes(), size):
                    subgraph_H = H.subgraph(nodes_H)
                    
                    if nx.is_isomorphic(subgraph_G, subgraph_H):
                        # Find the actual mapping
                        matcher = nx.algorithms.isomorphism.GraphMatcher(subgraph_G, subgraph_H)
                        if matcher.is_isomorphic():
                            mapping = matcher.mapping
                            return [(i, mapping[i]) for i in mapping]
        
        return []