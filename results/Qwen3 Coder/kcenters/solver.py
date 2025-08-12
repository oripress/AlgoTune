import networkx as nx
import bisect
from typing import Any, Dict, List, Tuple, Set
from pysat.solvers import Solver as SATSolver
import heapq

class Solver:
    def solve(self, problem: Tuple[Dict[str, Dict[str, float]], int], **kwargs) -> Any:
        """Solve the k-center problem optimally using binary search and SAT solving."""
        G_dict, k = problem

        # Handle edge cases
        if k <= 0:
            return []
            
        # Get all nodes
        nodes = list(G_dict.keys())
        if not nodes:
            return []
            
        # If k is greater than or equal to the number of nodes, return all nodes
        if k >= len(nodes):
            return nodes

        # Build networkx graph
        graph = nx.Graph()
        for v, adj in G_dict.items():
            for w, d in adj.items():
                graph.add_edge(v, w, weight=d)

        # Compute all-pairs shortest paths
        all_distances = dict(nx.all_pairs_dijkstra_path_length(graph))
        
        # First, get a greedy solution as an upper bound
        greedy_solution = self._greedy_heuristic(nodes, all_distances, k)
        greedy_obj = self._compute_max_distance(greedy_solution, all_distances, nodes)
        
        # Get all unique distances and sort them
        distances_set = set()
        for u in nodes:
            for v in nodes:
                if u in all_distances and v in all_distances[u]:
                    distances_set.add(all_distances[u][v])
        
        distances = sorted([d for d in distances_set if d <= greedy_obj])
        
        if not distances:
            return greedy_solution
            
        # Binary search on the objective value
        left, right = 0, len(distances) - 1
        best_solution = greedy_solution
        
        # Create node to variable mapping for SAT solver
        node_vars = {node: i for i, node in enumerate(nodes, start=1)}
        
        while left <= right:
            mid = (left + right) // 2
            radius = distances[mid]
            
            # Create SAT solver instance
            solver = SATSolver("MiniCard")
            
            # Add constraint: at most k centers
            solver.add_atmost(list(node_vars.values()), k=k)
            
            # Add constraint: each node must be within radius of some center
            for v in nodes:
                clause = []
                for u in nodes:
                    if all_distances[v][u] <= radius:
                        clause.append(node_vars[u])
                solver.add_clause(clause)
            
            # Check if satisfiable
            if solver.solve():
                # Solution exists with this radius
                model = solver.get_model()
                best_solution = [node for node, var in node_vars.items() if var in model]
                right = mid - 1  # Try to find a smaller radius
            else:
                left = mid + 1   # Need a larger radius
            
            solver.delete()
        
        return best_solution
    
    def _greedy_heuristic(self, nodes, all_distances, k):
        """Greedy farthest-point heuristic."""
        if k <= 0:
            return []
        
        if k >= len(nodes):
            return nodes[:]
            
        # Choose first center to minimize maximum distance
        first_center = min(
            nodes,
            key=lambda c: max(all_distances[c].get(u, float('inf')) for u in nodes)
        )
        centers = [first_center]
        
        # Choose remaining centers using farthest-point heuristic
        for _ in range(1, k):
            # Find the node that is farthest from any current center
            farthest_node = max(
                (node for node in nodes if node not in centers),
                key=lambda node: min(all_distances[node][center] for center in centers)
            )
            centers.append(farthest_node)
        
        return centers
    
    def _compute_max_distance(self, centers, all_distances, nodes):
        """Compute the maximum distance from any node to its nearest center."""
        max_dist = 0
        for node in nodes:
            min_dist = min(all_distances[node][center] for center in centers)
            max_dist = max(max_dist, min_dist)
        return max_dist