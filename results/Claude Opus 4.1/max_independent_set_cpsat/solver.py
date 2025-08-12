import numpy as np
from ortools.sat.python import cp_model
from collections import deque

class Solver:
    def __init__(self):
        # Pre-configure solver for speed
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 1.0
        self.solver.parameters.num_search_workers = 4
        self.solver.parameters.search_branching = cp_model.FIXED_SEARCH
        
    def find_components(self, adj_matrix):
        """Find connected components using BFS."""
        n = len(adj_matrix)
        visited = [False] * n
        components = []
        
        for start in range(n):
            if not visited[start]:
                component = []
                queue = deque([start])
                visited[start] = True
                
                while queue:
                    node = queue.popleft()
                    component.append(node)
                    
                    for neighbor in range(n):
                        if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                
                components.append(component)
        
        return components
    
    def solve_component(self, adj_matrix, component):
        """Solve MIS for a single component."""
        n = len(component)
        
        # Special cases
        if n == 0:
            return []
        if n == 1:
            return [component[0]]
        if n == 2:
            if adj_matrix[component[0]][component[1]] == 0:
                return component
            else:
                return [component[0]]
        
        # Check if component is independent (no edges)
        has_edge = False
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[component[i]][component[j]] == 1:
                    has_edge = True
                    break
            if has_edge:
                break
        
        if not has_edge:
            return component
        
        # Use CP-SAT for larger components
        model = cp_model.CpModel()
        
        # Create variables
        nodes = {}
        for idx in component:
            nodes[idx] = model.NewBoolVar(f"x_{idx}")
        
        # Add constraints only for edges within component
        edge_count = 0
        for i, idx_i in enumerate(component):
            for j in range(i + 1, len(component)):
                idx_j = component[j]
                if adj_matrix[idx_i][idx_j] == 1:
                    model.Add(nodes[idx_i] + nodes[idx_j] <= 1)
                    edge_count += 1
        
        # If very sparse, might be bipartite or special structure
        if edge_count < n:
            # Try greedy first for very sparse graphs
            selected = self.greedy_mis(adj_matrix, component)
            if len(selected) >= n - edge_count:
                return selected
        
        # Objective
        model.Maximize(sum(nodes.values()))
        
        # Solve
        status = self.solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            selected = [idx for idx in component if self.solver.Value(nodes[idx]) == 1]
            return selected
        
        return []
    
    def greedy_mis(self, adj_matrix, nodes):
        """Fast greedy algorithm for MIS."""
        selected = []
        available = set(nodes)
        
        # Sort by degree (ascending)
        degrees = [(sum(adj_matrix[node][j] for j in nodes if j != node), node) for node in nodes]
        degrees.sort()
        
        for _, node in degrees:
            if node in available:
                selected.append(node)
                available.remove(node)
                # Remove neighbors
                for neighbor in nodes:
                    if neighbor in available and adj_matrix[node][neighbor] == 1:
                        available.remove(neighbor)
        
        return selected
    
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solves the max independent set problem.
        
        :param problem: A 2d adjacency matrix representing the graph.
        :return: A list of node indices included in the maximum independent set.
        """
        n = len(problem)
        
        # Handle trivial cases
        if n == 0:
            return []
        if n == 1:
            return [0]
        
        # Convert to numpy for faster operations
        adj_np = np.array(problem, dtype=np.int8)
        
        # Check if graph has no edges
        if np.sum(adj_np) == 0:
            return list(range(n))
        
        # Find connected components
        components = self.find_components(problem)
        
        # Solve each component independently
        result = []
        for component in components:
            if len(component) == 1:
                result.extend(component)
            else:
                result.extend(self.solve_component(problem, component))
        
        return sorted(result)