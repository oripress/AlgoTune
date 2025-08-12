import networkx as nx
from typing import Any, Dict, List, Set, Tuple
import bisect
from pysat.solvers import Solver as SATSolver

class Distances:
    """Helper class for managing distances in the graph."""
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._dist = dict(nx.all_pairs_dijkstra_path_length(graph))
        self._sorted_dists = None
    
    def dist(self, u: str, v: str) -> float:
        return self._dist[u][v]
    
    def all_vertices(self) -> List[str]:
        return list(self.graph.nodes())
    
    def vertices_in_range(self, v: str, limit: float) -> List[str]:
        return [u for u in self.graph.nodes() if self._dist[v][u] <= limit]
    
    def max_dist(self, centers: List[str]) -> float:
        if not centers:
            return float('inf')
        max_d = 0
        for v in self.graph.nodes():
            min_d = min(self._dist[c][v] for c in centers)
            max_d = max(max_d, min_d)
        return max_d
    
    def sorted_distances(self) -> List[float]:
        if self._sorted_dists is None:
            all_dists = set()
            for u in self.graph.nodes():
                for v in self.graph.nodes():
                    all_dists.add(self._dist[u][v])
            self._sorted_dists = sorted(all_dists)
        return self._sorted_dists

def solve(problem: tuple[dict[str, dict[str, float]], int]) -> list[str]:
    """Global solve function for validation."""
    solver = Solver()
    return solver.solve(problem)

def compute_objective(problem: tuple[dict[str, dict[str, float]], int], solution: list[str]) -> float:
    """Global compute_objective function for validation."""
    G_dict, k = problem
    graph = nx.Graph()
    for v, adj in G_dict.items():
        for w, d in adj.items():
            graph.add_edge(v, w, weight=d)
    distances = Distances(graph)
    return distances.max_dist(solution)

class Solver:
    def solve(self, problem: Tuple[Dict[str, Dict[str, float]], int]) -> Any:
        """
        Solves the k-centers problem using SAT-based optimization.
        """
        G_dict, k = problem
        
        # Handle edge cases
        if not G_dict or k == 0:
            return []
        
        # Build graph
        graph = nx.Graph()
        for v, adj in G_dict.items():
            for w, d in adj.items():
                graph.add_edge(v, w, weight=d)
        
        nodes = list(graph.nodes())
        n = len(nodes)
        
        if k >= n:
            return nodes
        
        # Use distances helper
        distances = Distances(graph)
        
        # First get a greedy solution
        centers = []
        remaining = set(nodes)
        
        # Pick first center that minimizes max distance
        if remaining:
            first_center = min(
                remaining,
                key=lambda c: max(distances.dist(c, u) for u in remaining)
            )
            centers.append(first_center)
            remaining.remove(first_center)
        
        # Greedily add centers
        while len(centers) < k and remaining:
            farthest_node = max(
                remaining,
                key=lambda v: min(distances.dist(c, v) for c in centers)
            )
            centers.append(farthest_node)
            remaining.remove(farthest_node)
        
        # Now optimize using SAT
        obj = distances.max_dist(centers)
        
        # Create SAT solver
        node_vars = {node: i+1 for i, node in enumerate(nodes)}
        sat_solver = SATSolver("MiniCard")
        sat_solver.add_atmost(list(node_vars.values()), k=k)
        
        # Get sorted distances for binary search
        sorted_dists = distances.sorted_distances()
        index = bisect.bisect_left(sorted_dists, obj)
        sorted_dists = sorted_dists[:index]
        
        if not sorted_dists:
            return centers
        
        # Try to improve solution
        best_centers = centers
        
        while sorted_dists:
            limit = sorted_dists.pop()
            
            # Create new SAT instance
            sat_solver = SATSolver("MiniCard")
            sat_solver.add_atmost(list(node_vars.values()), k=k)
            
            # Add distance constraints
            for v in nodes:
                clause = [node_vars[u] for u in distances.vertices_in_range(v, limit)]
                sat_solver.add_clause(clause)
            
            if sat_solver.solve():
                model = sat_solver.get_model()
                if model:
                    best_centers = [node for node, var in node_vars.items() if var in model]
                    obj = distances.max_dist(best_centers)
                    index = bisect.bisect_left(sorted_dists, obj)
                    sorted_dists = sorted_dists[:index]
            else:
                break
        
        return best_centers