import numpy as np
from ortools.sat.python import cp_model
from collections import deque

class Solver:
    def solve(self, problem, **kwargs):
        import numpy as np
        from ortools.sat.python import cp_model
        from collections import deque
        
        n = len(problem)
        if n == 0:
            return []
        
        # Build adjacency list
        adj_list = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if problem[i][j]:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
        
        # Preprocessing: remove degree-0 and degree-1 nodes
        active = [True] * n
        solution_set = []
        degree = [len(neighbors) for neighbors in adj_list]
        
        # Process degree-0 and degree-1 nodes with a queue
        q = deque()
        for i in range(n):
            if active[i] and degree[i] == 0:
                solution_set.append(i)
                active[i] = False
            elif active[i] and degree[i] == 1:
                q.append(i)
        
        while q:
            i = q.popleft()
            if not active[i]:
                continue
                
            # Add the degree-1 node to solution
            solution_set.append(i)
            active[i] = False
            
            # Find its active neighbor
            neighbor = -1
            for j in adj_list[i]:
                if active[j]:
                    neighbor = j
                    break
            
            if neighbor >= 0:
                # Remove the neighbor
                active[neighbor] = False
                # Update degrees of neighbors
                for k in adj_list[neighbor]:
                    if active[k]:
                        degree[k] -= 1
                        if degree[k] == 1:
                            q.append(k)
        
        # Collect remaining active nodes
        remaining_nodes = [i for i in range(n) if active[i]]
        if not remaining_nodes:
            return solution_set
        
        # Build active adjacency list
        active_adj_list = [[] for _ in range(n)]
        for i in remaining_nodes:
            for j in adj_list[i]:
                if active[j]:
                    active_adj_list[i].append(j)
        
        # Extract connected components
        visited = [False] * n
        components = []
        for node in remaining_nodes:
            if not visited[node]:
                comp = []
                stack = [node]
                visited[node] = True
                while stack:
                    current = stack.pop()
                    comp.append(current)
                    for neighbor in active_adj_list[current]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(neighbor)
                components.append(comp)
        
        # Solve each component independently
        for comp in components:
            comp_size = len(comp)
            if comp_size == 0:
                continue
                
            # Use brute-force for small components
            if comp_size <= 20:
                best_set = []
                best_size = -1
                # Iterate over all possible subsets
                for mask in range(1 << comp_size):
                    subset = []
                    for j in range(comp_size):
                        if mask & (1 << j):
                            subset.append(comp[j])
                    
                    # Check independence
                    valid = True
                    for i1 in range(len(subset)):
                        for i2 in range(i1+1, len(subset)):
                            if problem[subset[i1]][subset[i2]]:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid and len(subset) > best_size:
                        best_size = len(subset)
                        best_set = subset
                
                solution_set.extend(best_set)
            else:
                # Use CP-SAT for larger components
                model = cp_model.CpModel()
                node_vars = [model.NewBoolVar(f'x{i}') for i in comp]
                node_to_idx = {node: idx for idx, node in enumerate(comp)}
                
                # Add constraints for edges in the component
                for node in comp:
                    idx_i = node_to_idx[node]
                    for neighbor in active_adj_list[node]:
                        if neighbor in comp and neighbor > node:
                            idx_j = node_to_idx[neighbor]
                            model.Add(node_vars[idx_i] + node_vars[idx_j] <= 1)
                
                # Maximize the solution size
                model.Maximize(sum(node_vars))
                
                solver = cp_model.CpSolver()
                solver.parameters.num_search_workers = 4
                status = solver.Solve(model)
                
                if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                    for idx, node in enumerate(comp):
                        if solver.Value(node_vars[idx]):
                            solution_set.append(node)
        
        return solution_set