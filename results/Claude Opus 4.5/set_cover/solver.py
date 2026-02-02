from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]], **kwargs) -> list[int]:
        """
        Solves the set cover problem using CP-SAT solver with greedy hints.
        """
        m = len(problem)
        if m == 0:
            return []
        
        # Convert to sets
        subsets = [set(s) for s in problem]
        
        # Determine the universe
        universe = set()
        for s in subsets:
            universe.update(s)
        
        if not universe:
            return []
        
        # Build element -> subsets mapping
        element_to_subsets = {e: [] for e in universe}
        for i, s in enumerate(subsets):
            for e in s:
                element_to_subsets[e].append(i)
        
        # Greedy solution for hints
        greedy_selected = set()
        remaining = universe.copy()
        while remaining:
            best_idx = -1
            best_cover = 0
            for i in range(m):
                if i not in greedy_selected:
                    cover = len(subsets[i] & remaining)
                    if cover > best_cover:
                        best_cover = cover
                        best_idx = i
            if best_idx == -1:
                break
            greedy_selected.add(best_idx)
            remaining -= subsets[best_idx]
        
        # Create CP model
        model = cp_model.CpModel()
        
        # Binary variables
        x = [model.NewBoolVar(f'x_{i}') for i in range(m)]
        
        # Coverage constraints
        for e in universe:
            covering = element_to_subsets[e]
            if covering:
                model.AddBoolOr([x[i] for i in covering])
        
        # Objective
        model.Minimize(sum(x))
        
        # Add hints from greedy solution
        for i in range(m):
            model.AddHint(x[i], 1 if i in greedy_selected else 0)
        
        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i + 1 for i in range(m) if solver.Value(x[i]) == 1]
        
        return []