import itertools
import numpy as np

class Solver:
    def solve(self, problem):
        num_nodes, n = problem
        
        # Generate all possible vertices
        vertices = list(itertools.product(range(num_nodes), repeat=n))
        
        # Create a set of all vertices for quick lookup
        available = set(vertices)
        selected = []
        
        # Simple greedy approach: select vertices that don't conflict with already selected ones
        while available:
            # Pick the first available vertex
            vertex = available.pop()
            selected.append(vertex)
            
            # Remove all vertices that are adjacent to the selected vertex
            to_remove = set()
            for v in available:
                # Check if v is adjacent to vertex
                adjacent = True
                for i in range(n):
                    # In a cyclic graph, check if two vertices are adjacent
                    # Two vertices are adjacent if for all positions, the distance is <= 1
                    d = min((vertex[i] - v[i]) % num_nodes, (v[i] - vertex[i]) % num_nodes)
                    if d > 1:  # Not adjacent in the cyclic graph at this position
                        adjacent = False
                        break
                if adjacent:  # If they are adjacent in all positions, we need to remove v
                    to_remove.add(v)
            
            available -= to_remove
            
        return selected