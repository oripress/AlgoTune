from ortools.sat.python import cp_model
from typing import List, Any, Tuple

class Solver:
    def solve(self, problem: Tuple[int, List[List[int]], List[List[int]]], **kwargs) -> List[int]:
        """Optimized set cover with conflicts solver using greedy upper bound and parallel solving."""
        n, sets, conflicts = problem
        
        # Precompute covering sets for each object
        covering_sets = [[] for _ in range(n)]
        for i, s in enumerate(sets):
            for obj in s:
                if obj < n:  # Ensure valid object index
                    covering_sets[obj].append(i)
        
        # Precompute greedy solution for hint
        greedy_solution = self._greedy_set_cover(n, sets, conflicts)
        
        # If greedy solution is the trivial solution (size n), return immediately
        if len(greedy_solution) == n:
            return greedy_solution
            
        # Build CP-SAT model
        model = cp_model.CpModel()
        set_vars = [model.NewBoolVar(f"s_{i}") for i in range(len(sets))]
        
        # Coverage constraints
        for obj in range(n):
            model.Add(sum(set_vars[i] for i in covering_sets[obj]) >= 1)
        
        # Conflict constraints
        for conflict in conflicts:
            model.AddAtMostOne(set_vars[i] for i in conflict)
        
        # Objective: minimize number of sets
        # Objective: minimize number of sets
        model.Minimize(sum(set_vars))
        
        # Configure solver
        
        # Configure solver
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.max_time_in_seconds = 0.1  # Reduced from 0.5s for faster fallback
        
        # Set greedy solution as hint
        for i in range(len(sets)):
            model.AddHint(set_vars[i], 1 if i in greedy_solution else 0)
        
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(len(sets)) if solver.Value(set_vars[i])]
        else:
            return greedy_solution  # Fallback to greedy solution
    
    def _greedy_set_cover(self, n: int, sets: List[List[int]], conflicts: List[List[int]]) -> List[int]:
        """Optimized greedy set cover with conflict avoidance using priority queue."""
        import heapq
        # Precompute trivial set indices
        trivial_set_indices = [-1] * n
        for idx, s in enumerate(sets):
            if len(s) == 1:
                obj = s[0]
                if 0 <= obj < n:
                    trivial_set_indices[obj] = idx
        
        # Build conflict graph
        conflict_neighbors = [set() for _ in range(len(sets))]
        for conflict in conflicts:
            for i in range(len(conflict)):
                for j in range(i+1, len(conflict)):
                    u, v = conflict[i], conflict[j]
                    conflict_neighbors[u].add(v)
                    conflict_neighbors[v].add(u)
        
        # Precompute object coverage
        obj_covering_sets = [[] for _ in range(n)]
        for i, s in enumerate(sets):
            for obj in s:
                if 0 <= obj < n:
                    obj_covering_sets[obj].append(i)
        # Initialize data structures
        covered = set()
        selected = []
        remaining_objs = [set() for _ in range(len(sets))]
        for i, s in enumerate(sets):
            for obj in s:
                if 0 <= obj < n:
                    remaining_objs[i].add(obj)
        
        # Initialize priority queue
        heap = []
        for i in range(len(sets)):
            size = len(remaining_objs[i])
            heapq.heappush(heap, (-size, i))
        
        # Main greedy loop
        while heap and len(covered) < n:
            neg_gain, i = heapq.heappop(heap)
            current_gain = -neg_gain
            
            # Skip outdated or invalid entries
            if current_gain != len(remaining_objs[i]) or current_gain == 0:
                continue
            
            # Check for conflicts with selected sets
            if any(neighbor in selected for neighbor in conflict_neighbors[i]):
                continue
                
            # Select this set
            selected.append(i)
            new_covered = remaining_objs[i]
            covered |= new_covered
            remaining_objs[i] = set()  # Mark as processed
            
            # Update affected sets
            for obj in new_covered:
                for j in obj_covering_sets[obj]:
                    if j == i or j in selected:
                        continue
                    if obj in remaining_objs[j]:
                        remaining_objs[j].discard(obj)
                        new_size = len(remaining_objs[j])
                        heapq.heappush(heap, (-new_size, j))
        
        # Add trivial sets for uncovered objects
        uncovered = set(range(n)) - covered
        for obj in uncovered:
            if trivial_set_indices[obj] != -1:
                selected.append(trivial_set_indices[obj])
        
        return selected